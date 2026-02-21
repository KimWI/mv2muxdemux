import os, sys, math, subprocess, cv2, hashlib, warnings, argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from numba import njit, prange

# ==========================================================
# 1. Numba JIT 고속 연산부 (안전성 극대화)
# ==========================================================

@njit(fastmath=True, cache=True)
def _get_dist_sq(c1, c2):
    return (int(c1[0]) - int(c2[0]))**2 + (int(c1[1]) - int(c2[1]))**2 + (int(c1[2]) - int(c2[2]))**2

@njit(fastmath=True, cache=True)
def _apply_floyd_steinberg(img_array, pal_888):
    """오차 도미노 폭주를 막기 위해 변수별 명시적 클리핑 적용"""
    h, w = 192, 256
    temp_img = img_array.astype(np.float32)
    indices = np.zeros((h, w), dtype=np.int32)
    strength = 0.75 
    
    for y in range(h):
        for x in range(w):
            # 1. 수치 폭주 원천 차단 (하드 클리핑)
            r = temp_img[y, x, 0]
            g = temp_img[y, x, 1]
            b = temp_img[y, x, 2]
            r = 255.0 if r > 255.0 else (0.0 if r < 0.0 else r)
            g = 255.0 if g > 255.0 else (0.0 if g < 0.0 else g)
            b = 255.0 if b > 255.0 else (0.0 if b < 0.0 else b)
            
            # RGB 공간에서 나올 수 있는 최대 거리 한계치(약 195,075) 이상으로 설정
            min_d = 250000.0 
            best_i = 1
            
            # 15색 전체 대상 탐색 (블랙 앵커링 해제 반영)
            for p_i in range(1, 16):
                pr, pg, pb = pal_888[p_i, 0], pal_888[p_i, 1], pal_888[p_i, 2]
                d = (r - pr)**2 + (g - pg)**2 + (b - pb)**2
                if d < min_d:
                    min_d = d
                    best_i = p_i
            
            indices[y, x] = best_i
            
            # 2. 오차 계산 및 확산
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
# 2. 메인 엔지니어링 클래스 (Time-Seek 적용)
# ==========================================================

def parse_time_str(t_str):
    """FFmpeg 스타일의 시간 문자열을 초(float) 단위로 변환"""
    if not t_str: return 0.0
    try:
        parts = str(t_str).split(':')
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        elif len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0

class MV2ProEncoder:
    def __init__(self, input_video, output_mv2, quant_algo='kmeans', use_dither=False, start_time=None, end_time=None):
        self.input_video, self.output_mv2 = input_video, output_mv2
        self.quant_algo, self.use_dither = quant_algo.lower(), use_dither
        self.fps, self.width, self.height = 15, 256, 192 
        
        # 시간 처리
        self.start_sec = parse_time_str(start_time)
        self.end_sec = parse_time_str(end_time) if end_time else None

        base = os.path.splitext(os.path.basename(input_video))[0]
        seed = f"{input_video}_{os.getpid()}".encode()
        self.temp_mp3 = f"temp_{base}_{hashlib.md5(seed).hexdigest()[:8]}.mp3"

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
        print(f"[*] 오디오 구간 추출 (128kbps) | 시작: {self.start_sec}초 -> 종료: {self.end_sec if self.end_sec else '끝까지'}")
        
        # FFmpeg 구간 추출 명령 구성
        ffmpeg_cmd = ["ffmpeg", "-y"]
        if self.start_sec > 0:
            ffmpeg_cmd.extend(["-ss", str(self.start_sec)])
        if self.end_sec:
            ffmpeg_cmd.extend(["-to", str(self.end_sec)])
        ffmpeg_cmd.extend(["-i", self.input_video, "-vn", "-acodec", "libmp3lame", "-ac", "2", "-ar", "44100", "-b:a", "128k", self.temp_mp3])
        
        subprocess.run(ffmpeg_cmd, capture_output=True)
        
        with open(self.temp_mp3, "rb") as f: mp3_data = f.read()
        cap = cv2.VideoCapture(self.input_video)
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out_f = open(self.output_mv2, "wb")
        out_f.write(bytearray(b'MV2 ').ljust(512, b'\x00')) 

        idx, mp3_off, bps = 0, 0, 16000 
        
        while cap.isOpened():
            # 타임시크가 적용된 현재 프레임의 오리지널 시간 계산
            current_time = self.start_sec + (idx / 15.0)
            if self.end_sec and current_time > self.end_sec:
                break # 설정된 종료 시간에 도달하면 루프 종료
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_time * orig_fps))
            ret, frame = cap.read()
            if not ret or mp3_off >= len(mp3_data): break

            img_np = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((256,192), Image.LANCZOS))
            pal_333, pal_888_np = self._extract_palette(img_np)
            
            indices_map = _apply_floyd_steinberg(img_np, pal_888_np) if self.use_dither else np.zeros((192, 256), dtype=np.int32)

            if not self.use_dither:
                for y in range(192):
                    for x in range(256):
                        px = img_np[y, x]
                        min_d, best_i = 1e10, 1
                        for p_i in range(1, 16):
                            d = _get_dist_sq(px, pal_888_np[p_i])
                            if d < min_d: min_d, best_i = d, p_i
                        indices_map[y, x] = best_i

            pgt, ct = _encode_vram_numba_master(pal_888_np, indices_map)
            pal_b = bytearray()
            for r, g, b in pal_333: pal_b.extend([(r<<4)|b, g])

            block = bytearray(b'\x55' * 16384) 
            block[0:6144], block[6144:12288], block[12288:12318] = pgt.tobytes(), ct.tobytes(), pal_b
            
            target_a = int((idx + 1) * (bps / 15))
            sz = max(1, min(111, math.ceil((target_a - mp3_off) / 32))) 
            block[12800] = sz
            chunk = mp3_data[mp3_off : mp3_off + sz*32]
            block[12801 : 12801+len(chunk)] = chunk
            mp3_off += len(chunk)
            
            out_f.write(block)
            if idx % 30 == 0: print(f"  > {idx} 프레임 인코딩 중... ({current_time:.2f}s)")
            idx += 1

        eof = bytearray(16384); eof[12318] = 0x01; eof[12800] = 0x22 
        out_f.write(eof); cap.release(); out_f.close()
        if os.path.exists(self.temp_mp3): os.remove(self.temp_mp3)
        print(f"[!] 완료: {self.output_mv2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSX2 MV2 Advanced Encoder")
    parser.add_argument("input", help="입력 동영상 파일 (.mp4)")
    parser.add_argument("output", help="출력 동영상 파일 (.mv2)")
    parser.add_argument("--algo", choices=['kmeans', 'mediancut', 'octree'], default='mediancut', help="색상 추출 알고리즘")
    parser.add_argument("--dither", action="store_true", help="Floyd-Steinberg 디더링 활성화")
    parser.add_argument("-ss", dest="start", help="시작 시간 (예: 00:00:15 또는 15)", default=None)
    parser.add_argument("-to", dest="end", help="종료 시간 (예: 00:00:30 또는 30)", default=None)
    
    args = parser.parse_args()
    
    MV2ProEncoder(
        input_video=args.input, 
        output_mv2=args.output, 
        quant_algo=args.algo, 
        use_dither=args.dither,
        start_time=args.start,
        end_time=args.end
    ).run()
