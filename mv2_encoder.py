import os
import sys
import math
import subprocess
import cv2
import numpy as np
import warnings
import hashlib
from PIL import Image
from sklearn.cluster import KMeans

class MV2MasterEncoder:
    def __init__(self, input_video, output_mv2, fps=15, quant_algo='mediancut'):
        self.input_video = input_video
        self.output_mv2 = output_mv2
        self.fps = fps
        self.width = 256
        self.height = 192
        self.quant_algo = quant_algo.lower()
        
        # [충돌 방지] 프로세스 ID와 파일명 해시를 조합한 고유 임시 파일명 생성
        base_name = os.path.splitext(os.path.basename(input_video))[0]
        unique_seed = f"{input_video}_{os.getpid()}".encode('utf-8')
        hash_str = hashlib.md5(unique_seed).hexdigest()[:8]
        self.temp_mp3 = f"temp_audio_{base_name}_{hash_str}.mp3"

    def _rgb888_to_333(self, rgb):
        """MSX2 9비트 팔레트 규격으로 변환"""
        return tuple(int(round((c / 255.0) * 7)) for c in rgb)

    def _rgb333_to_888(self, rgb333):
        """거리 계산을 위해 888로 복원"""
        return tuple(int(c * 255 // 7) for c in rgb333)

    def _color_dist(self, c1, c2):
        """uint8 오버플로우 방지 처리된 유클리드 거리 계산"""
        return (int(c1[0]) - int(c2[0]))**2 + \
               (int(c1[1]) - int(c2[1]))**2 + \
               (int(c1[2]) - int(c2[2]))**2

    def _extract_palette(self, img_array):
        """고급 양자화: 엣지 가중치 + 컬러 앵커링 적용"""
        pixels = img_array.reshape(-1, 3)
        
        # 엣지 가중치 부여 (윤곽선 보호)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = img_array[edges == 255]
        
        if len(edge_pixels) > 0:
            weighted_pixels = np.vstack([pixels] + [edge_pixels] * 5)
        else:
            weighted_pixels = pixels

        target_colors = 14 # 1번 슬롯(검은색) 제외
        raw_pal = []
        
        if self.quant_algo == 'kmeans':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=target_colors, n_init=1, max_iter=10, random_state=42).fit(weighted_pixels)
                raw_pal = [tuple(c) for c in kmeans.cluster_centers_]
        else:
            aug_img = Image.fromarray(weighted_pixels.reshape(1, -1, 3).astype(np.uint8))
            method = Image.Quantize.MEDIANCUT if self.quant_algo == 'mediancut' else Image.Quantize.FASTOCTREE
            q_img = aug_img.quantize(colors=target_colors, method=method)
            p_data = q_img.getpalette()
            raw_pal = [tuple(p_data[i:i+3]) for i in range(0, min(target_colors * 3, len(p_data)), 3)] if p_data else []

        # 컬러 앵커링: 순수 검은색 강제 삽입 및 유사 어두운 색 필터링
        filtered_pal = [c for c in raw_pal if sum(c) > 45]
        final_pal = [(0, 0, 0)] + filtered_pal
        while len(final_pal) < 15: final_pal.append((0, 0, 0))

        return [self._rgb888_to_333(c) for c in final_pal[:15]]

    def encode_vram_block(self, img_array):
        """MSX2 SCREEN 2 규격(8x1 Color Clash) 인코딩"""
        pal_333 = self._extract_palette(img_array)
        pal_888 = [(0,0,0)] + [self._rgb333_to_888(p) for p in pal_333]
        
        pgt, ct = bytearray(6144), bytearray(6144)
        
        for y in range(self.height):
            for cx in range(32):
                block_pixels = img_array[y, cx*8 : cx*8+8]
                mapped = [np.argmin([self._color_dist(px, p) for p in pal_888[1:16]]) + 1 for px in block_pixels]
                
                unique, counts = np.unique(mapped, return_counts=True)
                sorted_c = unique[np.argsort(-counts)]
                fg = sorted_c[0]
                bg = sorted_c[1] if len(sorted_c) > 1 else fg
                
                p_byte = 0
                for i, idx in enumerate(mapped):
                    if self._color_dist(pal_888[idx], pal_888[fg]) <= self._color_dist(pal_888[idx], pal_888[bg]):
                        p_byte |= (1 << (7 - i))
                
                off = ((y // 8) * 32 + cx) * 8 + (y % 8)
                pgt[off], ct[off] = p_byte, (fg << 4) | bg

        pal_bytes = bytearray()
        for r, g, b in pal_333:
            pal_bytes.extend([(r << 4) | b, g])
            
        return pgt, ct, pal_bytes

    def run(self):
        print(f"[*] 오디오 추출 중: {self.temp_mp3}")
        subprocess.run(["ffmpeg", "-y", "-i", self.input_video, "-vn", "-acodec", "libmp3lame", 
                        "-ac", "2", "-ar", "44100", "-b:a", "128k", self.temp_mp3], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(self.temp_mp3, "rb") as f: mp3_data = f.read()

        cap = cv2.VideoCapture(self.input_video)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out_f = open(self.output_mv2, "wb")
        out_f.write(bytearray(b'MV2 ').ljust(512, b'\x00')) # 헤더

        mp3_off, frame_idx = 0, 0
        bps = 128000 // 8

        print(f"[*] 인코딩 시작 (알고리즘: {self.quant_algo.upper()})")
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx * (orig_fps / self.fps)))
            ret, frame = cap.read()
            if not ret or mp3_off >= len(mp3_data): break

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((self.width, self.height), Image.LANCZOS)
            pgt, ct, pal = self.encode_vram_block(np.array(img))
            
            block = bytearray(b'\x55' * (15872 if frame_idx == 0 else 16384))
            block[0:6144], block[6144:12288], block[12288:12318] = pgt, ct, pal
            
            target_a = int((frame_idx + 1) * (bps / self.fps))
            sz_ind = max(1, min(128, math.ceil((target_a - mp3_off) / 32)))
            block[12800] = sz_ind
            
            chunk = mp3_data[mp3_off : mp3_off + sz_ind * 32].ljust(sz_ind * 32, b'\x55')
            block[12801 : 12801 + len(chunk)] = chunk
            mp3_off += len(chunk)
            
            out_f.write(block)
            if frame_idx % 30 == 0: print(f"  > {frame_idx} 프레임 완료...")
            frame_idx += 1

        # [실기 호환 EOF 블록]
        eof = bytearray(16384)
        eof[12318], eof[12800] = 0x01, 0x22
        for i in range(9): eof[12320 + i] = 0x0F
        out_f.write(eof)
        
        cap.release(); out_f.close()
        if os.path.exists(self.temp_mp3): os.remove(self.temp_mp3)
        print(f"[!] 생성 완료: {self.output_mv2}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: uv run mv2_encoder.py [input.mp4] [output.mv2] [kmeans/mediancut]")
        sys.exit(1)
    MV2MasterEncoder(sys.argv[1], sys.argv[2], quant_algo=(sys.argv[3] if len(sys.argv)>3 else 'mediancut')).run()
