import os
import sys
import math
import subprocess
import cv2
import numpy as np
import warnings
import hashlib  # 💡 해시 생성을 위해 추가
from PIL import Image
from sklearn.cluster import KMeans

class MV2MasterEncoder:
    #def __init__(self, input_video, output_mv2, fps=15, quant_algo='mediancut'):
    #    self.input_video = input_video
    #    self.output_mv2 = output_mv2
    #    self.fps = fps
    #    self.width = 256
    #    self.height = 192
    #    self.temp_mp3 = "advan_temp_audio.mp3"
    #    self.quant_algo = quant_algo.lower()

    def __init__(self, input_video, output_mv2, fps=15, quant_algo='mediancut'):
        self.input_video = input_video
        self.output_mv2 = output_mv2
        self.fps = fps
        self.width = 256
        self.height = 192
        self.quant_algo = quant_algo.lower()
        
        # 💡 [핵심] 임시 오디오 파일 충돌 방지 로직
        # 원본 파일의 순수 이름 추출 (예: SSF2T.mp4 -> SSF2T)
        base_name = os.path.splitext(os.path.basename(input_video))[0]
        
        # 절대 겹치지 않도록 '전체 경로 + 현재 실행 중인 프로세스 ID(PID)'를 시드로 사용
        unique_seed = f"{input_video}_{os.getpid()}".encode('utf-8')
        hash_str = hashlib.md5(unique_seed).hexdigest()[:8] # 8자리 짧은 해시
        
        # 결과물 예시: temp_audio_SSF2T_a1b2c3d4.mp3
        self.temp_mp3 = f"temp_audio_{base_name}_{hash_str}.mp3"
    # ==========================================================
    # 1. 컬러 유틸리티 (오버플로우 방지 적용)
    # ==========================================================
    def _rgb888_to_333(self, rgb):
        r, g, b = [int(round((c / 255.0) * 7)) for c in rgb]
        return (r, g, b)

    def _rgb333_to_888(self, rgb333):
        r, g, b = rgb333
        return (r * 255 // 7, g * 255 // 7, b * 255 // 7)

    def _color_dist(self, c1, c2):
        return (int(c1[0]) - int(c2[0]))**2 + \
               (int(c1[1]) - int(c2[1]))**2 + \
               (int(c1[2]) - int(c2[2]))**2

    # ==========================================================
    # 2. 고급 팔레트 추출 (Edge-Weighted + Anchoring)
    # ==========================================================
    def _extract_palette(self, img_array):
        pixels = img_array.reshape(-1, 3)
        
        # [핵심 1] OpenCV Canny 엣지 검출을 이용해 경계선 픽셀 추출
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_pixels = img_array[edges == 255]
        
        # 경계선 픽셀에 5배의 가중치를 부여하여 배열을 뻥튀기함
        if len(edge_pixels) > 0:
            weighted_pixels = np.vstack([pixels] + [edge_pixels] * 5)
        else:
            weighted_pixels = pixels

        unique_colors = np.unique(weighted_pixels, axis=0)
        
        # 1번 팔레트는 무조건 '순수 검은색'으로 고정할 것이므로, 알고리즘은 14개만 찾도록 지시
        target_colors = 14
        raw_pal = []
        
        if len(unique_colors) <= target_colors:
            raw_pal = [tuple(c) for c in unique_colors]
            
        elif self.quant_algo == 'kmeans':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 가중치가 적용된 데이터셋으로 K-Means 수행
                kmeans = KMeans(n_clusters=target_colors, n_init=1, max_iter=10, random_state=42).fit(weighted_pixels)
                raw_pal = [tuple(c) for c in kmeans.cluster_centers_]
                
        else:
            # Pillow의 C-Engine을 속이기 위해, 가중치가 적용된 1D 픽셀 배열을 (1, 길이, 3) 형태의 얇고 긴 이미지로 변조
            aug_img_array = weighted_pixels.reshape(1, -1, 3).astype(np.uint8)
            img = Image.fromarray(aug_img_array)
            method = Image.Quantize.MEDIANCUT if self.quant_algo == 'mediancut' else Image.Quantize.FASTOCTREE
            
            q_img = img.quantize(colors=target_colors, method=method)
            pal_data = q_img.getpalette()
            
            if pal_data:
                for i in range(0, min(target_colors * 3, len(pal_data)), 3):
                    raw_pal.append((pal_data[i], pal_data[i+1], pal_data[i+2]))

        # [핵심 2] 어중간한 탁한 색상(다크 그레이, 다크 블루 등) 제거
        # 순수 검은색(0,0,0)과 역할이 겹쳐 팔레트를 낭비하는 것을 막습니다.
        filtered_pal = [c for c in raw_pal if sum(c) > 45]

        # [핵심 3] 컬러 앵커링: 무조건 첫 번째 슬롯에 순수 검은색을 강제 박제
        final_pal = [(0, 0, 0)] + filtered_pal

        # 15개가 안 채워졌다면 빈 공간을 검은색으로 마저 채움
        while len(final_pal) < 15:
            final_pal.append((0, 0, 0))

        # MSX2 하드웨어 컬러(333) 규격으로 양자화
        pal_333 = [self._rgb888_to_333(c) for c in final_pal[:15]]
        return pal_333

    # ==========================================================
    # 3. MSX2 VRAM 변환 (Color Clash 제어)
    # ==========================================================
    def encode_vram_block(self, img_array):
        pal_333 = self._extract_palette(img_array)
        pal_888 = [(0,0,0)] + [self._rgb333_to_888(p) for p in pal_333]
        
        pgt = bytearray(6144)
        ct = bytearray(6144)
        
        for y in range(self.height):
            for cx in range(32):
                x_start = cx * 8
                block_pixels = img_array[y, x_start:x_start+8]
                
                mapped = []
                for px in block_pixels:
                    dists = [self._color_dist(px, p) for p in pal_888[1:16]]
                    mapped.append(np.argmin(dists) + 1)
                
                unique, counts = np.unique(mapped, return_counts=True)
                sorted_colors = unique[np.argsort(-counts)]
                fg = sorted_colors[0]
                bg = sorted_colors[1] if len(sorted_colors) > 1 else fg
                
                p_byte = 0
                for i in range(8):
                    px_idx = mapped[i]
                    if self._color_dist(pal_888[px_idx], pal_888[fg]) <= self._color_dist(pal_888[px_idx], pal_888[bg]):
                        p_byte |= (1 << (7 - i))
                        
                b_idx = (y // 8) * 32 + cx
                off = (b_idx * 8) + (y % 8)
                pgt[off] = p_byte
                ct[off] = (fg << 4) | bg

        pal_bytes = bytearray(30)
        for i in range(15):
            r, g, b = pal_333[i]
            pal_bytes[i*2] = (r << 4) | b
            pal_bytes[i*2 + 1] = g
            
        return pgt, ct, pal_bytes

    # ==========================================================
    # 4. Muxing 및 조립 파이프라인
    # ==========================================================
    def run(self):
        print(f"[*] 1. FFmpeg으로 오디오 추출 중 (128kbps MP3)...")
        subprocess.run([
            "ffmpeg", "-y", "-i", self.input_video, 
            "-vn", "-acodec", "libmp3lame", "-ac", "2", "-ar", "44100", "-b:a", "128k", 
            self.temp_mp3
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(self.temp_mp3, "rb") as f:
            mp3_data = f.read()

        print(f"[*] 2. 비디오 분석 및 MV2 Muxing 시작 (알고리즘: {self.quant_algo.upper()})...")
        cap = cv2.VideoCapture(self.input_video)
        
        if not cap.isOpened():
            print("❌ 동영상 파일을 열 수 없습니다.")
            return

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps == 0 or math.isnan(orig_fps):
            orig_fps = 30.0
            
        out_f = open(self.output_mv2, "wb")
        
        header = bytearray(512)
        header[0:4] = b'MV2 '
        out_f.write(header)

        mp3_offset = 0
        audio_bytes_per_sec = 128000 // 8
        frame_idx = 0

        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx * (orig_fps / self.fps)))
            ret, frame = cap.read()
            if not ret or mp3_offset >= len(mp3_data):
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((self.width, self.height), Image.Resampling.LANCZOS)
            
            pgt, ct, pal = self.encode_vram_block(np.array(img))
            
            payload_size = 15872 if frame_idx == 0 else 16384
            block = bytearray(b'\x55' * payload_size)
            
            block[0:6144] = pgt
            block[6144:12288] = ct
            block[12288:12318] = pal
            
            target_audio = int((frame_idx + 1) * (audio_bytes_per_sec / self.fps))
            bytes_needed = target_audio - mp3_offset
            
            size_indicator = math.ceil(bytes_needed / 32)
            size_indicator = max(1, min(128, size_indicator))
            chunk_size = size_indicator * 32
            
            block[12800] = size_indicator
            
            audio_chunk = mp3_data[mp3_offset : mp3_offset + chunk_size]
            if len(audio_chunk) < chunk_size:
                audio_chunk += b'\x55' * (chunk_size - len(audio_chunk))
            
            block[12801 : 12801 + chunk_size] = audio_chunk
            mp3_offset += chunk_size
            
            out_f.write(block)
            
            if frame_idx % 30 == 0:
                print(f"  > 인코딩 진행 중... {frame_idx} 프레임 완료")
            frame_idx += 1

        cap.release()
        
        # [핵심 4] 역공학으로 알아낸 오리지널 AVGEN의 완벽한 EOF 시그니처 블록 생성
        print("[*] 3. MSX 실기 정상 종료를 위한 오리지널 EOF 플래그 추가 중...")
        eof_block = bytearray(16384) 
        eof_block[12318] = 0x01  # 실기 플레이어 종료 플래그
        for i in range(9):
            eof_block[12320 + i] = 0x0F
        eof_block[12800] = 0x22
        
        out_f.write(eof_block)
        out_f.close()
        
        if os.path.exists(self.temp_mp3): os.remove(self.temp_mp3)
        print(f"[!] 완벽한 MSX2 MV2 파일 생성 완료: {self.output_mv2}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python mv2_encoder_advanced.py [입력.mp4] [출력.mv2] [알고리즘: kmeans/mediancut/octree]")
        sys.exit(1)
    
    algo = sys.argv[3] if len(sys.argv) > 3 else 'mediancut'
    print(f"[*] 선택된 양자화 알고리즘: {algo.upper()}")
    
    encoder = MV2MasterEncoder(sys.argv[1], sys.argv[2], quant_algo=algo)
    encoder.run()
