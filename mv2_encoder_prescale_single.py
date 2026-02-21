import os, sys, math, subprocess, cv2, hashlib, warnings, argparse
import numpy as np
from sklearn.cluster import KMeans
from numba import njit, prange

# ==========================================================
# 1. 시간 파싱 유틸리티
# ==========================================================
def parse_time_str(t_str):
    if not t_str: return 0.0
    try:
        parts = str(t_str).split(':')
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        elif len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0

# ==========================================================
# 2. Numba JIT 고속 연산부
# ==========================================================
@njit(fastmath=True, cache=True)
def _get_dist_sq(c1, c2):
    return (int(c1[0]) - int(c2[0]))**2 + (int(c1[1]) - int(c2[1]))**2 + (int(c1[2]) - int(c2[2]))**2

@njit(fastmath=True, cache=True)
def _apply_floyd_steinberg(img_array, pal_888):
    h, w = 192, 256
    temp_img = img_array.astype(np.float32)
    indices = np.zeros((h, w), dtype=np.int32)
    strength = 0.75 
    
    for y in range(h):
        for x in range(w):
            r = temp_img[y, x, 0]
            g = temp_img[y, x, 1]
            b = temp_img[y, x, 2]
            r = 255.0 if r > 255.0 else (0.0 if r < 0.0 else r)
            g = 255.0 if g > 255.0 else (0.0 if g < 0.0 else g)
            b = 255.0 if b > 255.0 else (0.0 if b < 0.0 else b)
            
            min_d = 250000.0 
            best_i = 1
            for p_i in range(1, 16):
                pr, pg, pb = pal_888[p_i, 0], pal_888[p_i, 1], pal_888[p_i, 2]
                d = (r - pr)**2 + (g - pg)**2 + (b - pb)**2
                if d < min_d:
                    min_d = d
                    best_i = p_i
            
            indices[y, x] = best_i
            nr, ng, nb = pal_888[best_i, 0], pal_888[best_i, 1], pal_888[best_i, 2]
            er = (r - nr) * strength
            eg = (g - ng) * strength
            eb = (b - nb) * strength
            
            if x + 1 < w:
                temp_img[y, x+1, 0] += er * 0.4375
                temp_img[y, x+1, 1] += eg * 0.4375
                temp_img[y, x+1, 2] += eb * 0.4375
            if y + 1 < h:
                if x > 0:
                    temp_img[y+1, x-1, 0] += er * 0.1875
                    temp_img[y+1, x-1, 1] += eg * 0.1875
                    temp_img[y+1, x-1, 2] += eb * 0.1875
                temp_img[y+1, x, 0] += er * 0.3125
                temp_img[y+1, x, 1] += eg * 0.3125
                temp_img[y+1, x, 2] += eb * 0.3125
                if x + 1 < w:
                    temp_img[y+1, x+1, 0] += er * 0.0625
                    temp_img[y+1, x+1, 1] += eg * 0.0625
                    temp_img[y+1, x+1, 2] += eb * 0.0625
    return indices

@njit(parallel=True, fastmath=True, cache=True)
def _encode_vram_numba_master(pal_888, indices_map):
    h, w = 192, 256
    pgt, ct = np.zeros(6144, dtype=np.uint8), np.zeros(6144, dtype=np.uint8)
    
    for y in prange(h):
        for cx in range(32):
            x_start = cx * 8
            block_indices = indices_map[y, x_start : x_start + 8]
            
            counts = np.zeros(16, dtype=np.int32)
            for idx in block_indices: 
                if 1 <= idx <= 15: counts[idx] += 1
            
            fg, bg = 1, 1
            m1, m2 = -1, -1
            for i in range(1, 16):
                if counts[i] > m1: m2, bg, m1, fg = m1, bg, counts[i], i
                elif counts[i] > m2: m2, bg = counts[i], i
            
            if fg == bg:
                max_d = -1.0
                for i in range(1, 16):
                    d = _get_dist_sq(pal_888[fg], pal_888[i])
                    if d > max_d: max_d, bg = d, i

            p_byte = 0
            for i in range(8):
                px_idx = block_indices[i]
                if _get_dist_sq(pal_888[px_idx], pal_888[fg]) <= _get_dist_sq(pal_888[px_idx], pal_888[bg]):
                    p_byte |= (1 << (7 - i))
            
            off = ((y // 8) * 32 + cx) * 8 + (y % 8)
            pgt[off], ct[off] = p_byte, (fg << 4) | bg
    return pgt, ct

# ==========================================================
# 3. 메인 인코더 클래스
# ==========================================================
class MV2SuperSamplerEncoder:
    def __init__(self, input_video, output_mv2, quant_algo='kmeans', use_dither=False, start_time=None, end_time=None, aspect_mode='pad', skip_prescale=False):
        self.input_video = input_video
        self.output_mv2 = output_mv2
        self.quant_algo = quant_algo.lower()
        self.use_dither = use_dither
        self.aspect_mode = aspect_mode.lower()
        self.skip_prescale = skip_prescale
        
        self.start_sec = parse_time_str(start_time)
        self.end_sec = parse_time_str(end_time) if end_time else None

        base = os.path.splitext(os.path.basename(input_video))[0]
        seed = f"{input_video}_{os.getpid()}".encode()
        hash_str = hashlib.md5(seed).hexdigest()[:8]
        
        self.temp_mp3 = f"temp_audio_{base}_{hash_str}.mp3"
        self.temp_vid = f"temp_video_{base}_{hash_str}.mp4"

    def _extract_palette(self, img_np):
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_px = img_np[edges == 255]
        weighted = np.vstack([img_np.reshape(-1,3)] + [edge_px]*5) if len(edge_px)>0 else img_np.reshape(-1,3)
        
        n_colors = min(len(np.unique(weighted, axis=0)), 15)
        if n_colors < 1:
            raw = [(0,0,0)] * 15
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=n_colors, n_init=1, max_iter=10).fit(weighted)
                raw = [tuple(c) for c in km.cluster_centers_]

        final_pal_888 = raw
        while len(final_pal_888) < 15: final_pal_888.append((0,0,0))
        
        pal_333 = [tuple(int(round((c/255.0)*7)) for c in rgb) for rgb in final_pal_888[:15]]
        pal_888_np = np.zeros((16, 3), dtype=np.int32)
        for i, p in enumerate(pal_333):
            pal_888_np[i+1] = [int(c*255//7) for c in p]
        return pal_333, pal_888_np

    def run(self):
        print(f"[*] 인코딩 시작 (Pre-scale 스킵: {self.skip_prescale}, 비율: {self.aspect_mode.upper()})")
        
        time_args = []
        if self.start_sec > 0: time_args.extend(["-ss", str(self.start_sec)])
        if self.end_sec: time_args.extend(["-to", str(self.end_sec)])

        # 1. 오디오 추출 (id3 제거 적용)
        cmd_audio = ["ffmpeg", "-y"] + time_args + ["-i", self.input_video, 
                     "-vn", "-acodec", "libmp3lame", "-ac", "2", "-ar", "44100", 
                     "-b:a", "128k", "-id3v2_version", "0", self.temp_mp3]
        subprocess.run(cmd_audio, capture_output=True)

        # 2. 비디오 소스 결정
        if not self.skip_prescale:
            print("[*] FFmpeg 512x384 슈퍼샘플링 사전 렌더링 중...")
            if self.aspect_mode == 'pad':
                vf_string = "scale=512:384:force_original_aspect_ratio=decrease:flags=lanczos,pad=512:384:-1:-1:color=black"
            elif self.aspect_mode == 'crop':
                vf_string = "scale=512:384:force_original_aspect_ratio=increase:flags=lanczos,crop=512:384"
            else:
                vf_string = "scale=512:384:flags=lanczos"

            cmd_video = ["ffmpeg", "-y"] + time_args + ["-i", self.input_video, 
                         "-an", "-vf", vf_string, "-r", "15", 
                         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "10", self.temp_vid]
            subprocess.run(cmd_video, capture_output=True)
            
            cap = cv2.VideoCapture(self.temp_vid)
            orig_fps = 15.0 # 전처리 영상은 정확히 15fps
        else:
            print("[!] FFmpeg 비디오 전처리 생략. OpenCV 직접 읽기 모드 작동.")
            cap = cv2.VideoCapture(self.input_video)
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        with open(self.temp_mp3, "rb") as f: mp3_data = f.read()
        
        out_f = open(self.output_mv2, "wb")
        out_f.write(bytearray(b'MV2 ').ljust(512, b'\x00')) 

        idx, mp3_off, bps = 0, 0, 16000 
        
        while cap.isOpened():
            # OpenCV 직접 읽기 모드(-skip-prescale)일 경우에만 프레임 스킵 연산 적용
            if self.skip_prescale:
                current_time = self.start_sec + (idx / 15.0)
                if self.end_sec and current_time > self.end_sec:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_time * orig_fps))

            ret, frame = cap.read()
            if not ret or mp3_off >= len(mp3_data): break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 해상도 스케일링 분기
            if self.skip_prescale:
                img_512 = cv2.resize(img_rgb, (512, 384), interpolation=cv2.INTER_LANCZOS4)
            else:
                img_512 = img_rgb
            
            # 팔레트 추출 및 패턴 매핑
            pal_333, pal_888_np = self._extract_palette(img_512)
            img_256 = cv2.resize(img_512, (256, 192), interpolation=cv2.INTER_AREA)
            
            indices_map = _apply_floyd_steinberg(img_256, pal_888_np) if self.use_dither else np.zeros((192, 256), dtype=np.int32)

            if not self.use_dither:
                for y in range(192):
                    for x in range(256):
                        px = img_256[y, x]
                        min_d, best_i = 1e10, 1
                        for p_i in range(1, 16):
                            d = _get_dist_sq(px, pal_888_np[p_i])
                            if d < min_d: min_d, best_i = d, p_i
                        indices_map[y, x] = best_i

            pgt, ct = _encode_vram_numba_master(pal_888_np, indices_map)
            pal_b = bytearray()
            for r, g, b in pal_333: pal_b.extend([(r<<4)|b, g])

            # 🚨 첫 프레임 15872 바이트 헤더 오프셋 유지
            block = bytearray(b'\x55' * (15872 if idx == 0 else 16384)) 
            block[0:6144], block[6144:12288], block[12288:12318] = pgt.tobytes(), ct.tobytes(), pal_b
            
            target_a = int((idx + 1) * (bps / 15))
            sz = max(1, min(111, math.ceil((target_a - mp3_off) / 32))) 
            block[12800] = sz
            chunk = mp3_data[mp3_off : mp3_off + sz*32]
            block[12801 : 12801+len(chunk)] = chunk
            mp3_off += len(chunk)
            
            out_f.write(block)
            if idx % 30 == 0: print(f"  > {idx} 프레임 인코딩 중...")
            idx += 1

        eof = bytearray(16384); eof[12318] = 0x01; eof[12800] = 0x22 
        out_f.write(eof); cap.release(); out_f.close()
        
        # 정리
        if os.path.exists(self.temp_mp3): os.remove(self.temp_mp3)
        if os.path.exists(self.temp_vid): os.remove(self.temp_vid)
        
        print(f"[!] 인코딩 완료: {self.output_mv2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSX2 MV2 Ultimate Encoder")
    parser.add_argument("input", help="입력 동영상 파일 (.mp4)")
    parser.add_argument("output", help="출력 동영상 파일 (.mv2)")
    parser.add_argument("--algo", choices=['kmeans', 'mediancut', 'octree'], default='mediancut')
    parser.add_argument("--dither", action="store_true", help="오차 확산 디더링 켜기")
    parser.add_argument("-ss", dest="start", default=None, help="시작 시간 (예: 00:00:15)")
    parser.add_argument("-to", dest="end", default=None, help="종료 시간 (예: 00:00:30)")
    parser.add_argument("--aspect", choices=['pad', 'crop', 'force'], default='pad', help="화면 비율 유지 방식")
    parser.add_argument("--skip-prescale", action="store_true", help="FFmpeg 비디오 전처리 생략 (이미 가공된 영상일 경우)")
    
    args = parser.parse_args()
    
    MV2SuperSamplerEncoder(
        input_video=args.input, 
        output_mv2=args.output, 
        quant_algo=args.algo, 
        use_dither=args.dither, 
        start_time=args.start, 
        end_time=args.end, 
        aspect_mode=args.aspect, 
        skip_prescale=args.skip_prescale
    ).run()
