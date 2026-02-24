import os, sys, math, subprocess, cv2, hashlib, warnings, argparse
import numpy as np
from sklearn.cluster import KMeans
from numba import njit, prange
from PIL import Image

# ==========================================================
# 1. Numba JIT 고속 연산부 (단일 프레임 극한 최적화 및 역산)
# ==========================================================
@njit(fastmath=True, cache=True)
def _get_dist_sq(c1, c2):
    return (int(c1[0]) - int(c2[0]))**2 + (int(c1[1]) - int(c2[1]))**2 + (int(c1[2]) - int(c2[2]))**2

@njit(fastmath=True, cache=True)
def _apply_dither_rgb(img_array, pal_888, mode):
    """디더링 오차를 RGB 공간 자체에 확산시킵니다."""
    h, w = 192, 256
    temp_img = img_array.astype(np.float32)
    
    if mode == 0: # None
        return temp_img
        
    strength = 0.75 
    for y in range(h):
        for x in range(w):
            r = temp_img[y, x, 0]; g = temp_img[y, x, 1]; b = temp_img[y, x, 2]
            r = 255.0 if r > 255.0 else (0.0 if r < 0.0 else r)
            g = 255.0 if g > 255.0 else (0.0 if g < 0.0 else g)
            b = 255.0 if b > 255.0 else (0.0 if b < 0.0 else b)
            
            min_d = 250000.0; best_i = 1
            for p_i in range(1, 16):
                d = (r - pal_888[p_i,0])**2 + (g - pal_888[p_i,1])**2 + (b - pal_888[p_i,2])**2
                if d < min_d: min_d, best_i = d, p_i
            
            nr, ng, nb = pal_888[best_i, 0], pal_888[best_i, 1], pal_888[best_i, 2]
            er, eg, eb = (r - nr) * strength, (g - ng) * strength, (b - nb) * strength
            
            if mode == 1: # Floyd-Steinberg
                if x + 1 < w:
                    temp_img[y, x+1, 0] += er * 0.4375; temp_img[y, x+1, 1] += eg * 0.4375; temp_img[y, x+1, 2] += eb * 0.4375
                if y + 1 < h:
                    if x > 0:
                        temp_img[y+1, x-1, 0] += er * 0.1875; temp_img[y+1, x-1, 1] += eg * 0.1875; temp_img[y+1, x-1, 2] += eb * 0.1875
                    temp_img[y+1, x, 0] += er * 0.3125; temp_img[y+1, x, 1] += eg * 0.3125; temp_img[y+1, x, 2] += eb * 0.3125
                    if x + 1 < w:
                        temp_img[y+1, x+1, 0] += er * 0.0625; temp_img[y+1, x+1, 1] += eg * 0.0625; temp_img[y+1, x+1, 2] += eb * 0.0625
                        
            elif mode == 2: # Jarvis-Judice-Ninke
                if x + 1 < w: temp_img[y, x+1, 0] += er*(7/48); temp_img[y, x+1, 1] += eg*(7/48); temp_img[y, x+1, 2] += eb*(7/48)
                if x + 2 < w: temp_img[y, x+2, 0] += er*(5/48); temp_img[y, x+2, 1] += eg*(5/48); temp_img[y, x+2, 2] += eb*(5/48)
                if y + 1 < h:
                    if x - 2 >= 0: temp_img[y+1, x-2, 0] += er*(3/48); temp_img[y+1, x-2, 1] += eg*(3/48); temp_img[y+1, x-2, 2] += eb*(3/48)
                    if x - 1 >= 0: temp_img[y+1, x-1, 0] += er*(5/48); temp_img[y+1, x-1, 1] += eg*(5/48); temp_img[y+1, x-1, 2] += eb*(5/48)
                    temp_img[y+1, x, 0] += er*(7/48); temp_img[y+1, x, 1] += eg*(7/48); temp_img[y+1, x, 2] += eb*(7/48)
                    if x + 1 < w: temp_img[y+1, x+1, 0] += er*(5/48); temp_img[y+1, x+1, 1] += eg*(5/48); temp_img[y+1, x+1, 2] += eb*(5/48)
                    if x + 2 < w: temp_img[y+1, x+2, 0] += er*(3/48); temp_img[y+1, x+2, 1] += eg*(3/48); temp_img[y+1, x+2, 2] += eb*(3/48)
                if y + 2 < h:
                    if x - 2 >= 0: temp_img[y+2, x-2, 0] += er*(1/48); temp_img[y+2, x-2, 1] += eg*(1/48); temp_img[y+2, x-2, 2] += eb*(1/48)
                    if x - 1 >= 0: temp_img[y+2, x-1, 0] += er*(3/48); temp_img[y+2, x-1, 1] += eg*(3/48); temp_img[y+2, x-1, 2] += eb*(3/48)
                    temp_img[y+2, x, 0] += er*(5/48); temp_img[y+2, x, 1] += eg*(5/48); temp_img[y+2, x, 2] += eb*(5/48)
                    if x + 1 < w: temp_img[y+2, x+1, 0] += er*(3/48); temp_img[y+2, x+1, 1] += eg*(3/48); temp_img[y+2, x+1, 2] += eb*(3/48)
                    if x + 2 < w: temp_img[y+2, x+2, 0] += er*(1/48); temp_img[y+2, x+2, 1] += eg*(1/48); temp_img[y+2, x+2, 2] += eb*(1/48)
    return temp_img

@njit(parallel=True, fastmath=True, cache=True)
def _encode_vram_optimal_search(img_rgb_float, pal_888):
    """모든 15x15 색상 조합을 대입하여 8x1 블록의 MSE를 최소화하는 PGT/CT를 생성합니다."""
    h, w = 192, 256
    pgt, ct = np.zeros(6144, dtype=np.uint8), np.zeros(6144, dtype=np.uint8)
    
    for y in prange(h):
        for cx in range(32):
            x_start = cx * 8
            block_rgb = img_rgb_float[y, x_start : x_start + 8]
            
            best_err = 1e12
            best_fg, best_bg = 1, 1
            
            for i in range(1, 16):
                for j in range(1, i + 1):
                    err = 0.0
                    for p in range(8):
                        r, g, b = block_rgb[p]
                        r_cl, g_cl, b_cl = max(0.0, min(255.0, r)), max(0.0, min(255.0, g)), max(0.0, min(255.0, b))
                        
                        d_i = (r_cl - pal_888[i,0])**2 + (g_cl - pal_888[i,1])**2 + (b_cl - pal_888[i,2])**2
                        d_j = (r_cl - pal_888[j,0])**2 + (g_cl - pal_888[j,1])**2 + (b_cl - pal_888[j,2])**2
                        
                        err += d_i if d_i < d_j else d_j
                        
                    if err < best_err:
                        best_err = err
                        best_fg, best_bg = i, j
            
            p_byte = 0
            for p in range(8):
                r, g, b = block_rgb[p]
                r_cl, g_cl, b_cl = max(0.0, min(255.0, r)), max(0.0, min(255.0, g)), max(0.0, min(255.0, b))
                
                d_fg = (r_cl - pal_888[best_fg,0])**2 + (g_cl - pal_888[best_fg,1])**2 + (b_cl - pal_888[best_fg,2])**2
                d_bg = (r_cl - pal_888[best_bg,0])**2 + (g_cl - pal_888[best_bg,1])**2 + (b_cl - pal_888[best_bg,2])**2
                
                if d_fg <= d_bg:
                    p_byte |= (1 << (7 - p))
            
            off = ((y // 8) * 32 + cx) * 8 + (y % 8)
            pgt[off] = p_byte
            ct[off] = (best_fg << 4) | best_bg
            
    return pgt, ct

@njit(fastmath=True, cache=True)
def _reconstruct_msx_frame(pgt, ct, pal_888_np):
    """생성된 바이너리를 역산하여 실제 MSX2 출력 영상을 재구성합니다."""
    h, w = 192, 256
    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for cx in range(32):
            x_start = cx * 8
            off = ((y // 8) * 32 + cx) * 8 + (y % 8)
            p_byte = pgt[off]
            c_byte = ct[off]
            fg = c_byte >> 4
            bg = c_byte & 0x0F
            
            for p in range(8):
                if (p_byte & (1 << (7 - p))) != 0:
                    out_img[y, x_start + p, 0] = pal_888_np[fg, 0]
                    out_img[y, x_start + p, 1] = pal_888_np[fg, 1]
                    out_img[y, x_start + p, 2] = pal_888_np[fg, 2]
                else:
                    out_img[y, x_start + p, 0] = pal_888_np[bg, 0]
                    out_img[y, x_start + p, 1] = pal_888_np[bg, 1]
                    out_img[y, x_start + p, 2] = pal_888_np[bg, 2]
    return out_img

# ==========================================================
# 2. 메인 오케스트레이터 클래스
# ==========================================================
def parse_time_str(t_str):
    if not t_str: return 0.0
    try:
        parts = str(t_str).split(':')
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        elif len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(parts[0])
    except ValueError: return 0.0

class MV2PerfectFrameEncoder:
    def __init__(self, input_video, output_mv2, quant_algo='kmeans', dither_mode='none', start_time=None, end_time=None, aspect_mode='pad', skip_prescale=False, debug_frames=False):
        self.input_video = input_video
        self.output_mv2 = output_mv2
        self.quant_algo = quant_algo.lower()
        self.dither_mode = dither_mode.lower()
        self.aspect_mode = aspect_mode.lower()
        self.skip_prescale = skip_prescale
        self.debug_frames = debug_frames
        self.start_sec = parse_time_str(start_time)
        self.end_sec = parse_time_str(end_time) if end_time else None

        self.base_name = os.path.splitext(os.path.basename(input_video))[0]
        hash_str = hashlib.md5(f"{input_video}_{os.getpid()}".encode()).hexdigest()[:8]
        self.temp_mp3 = f"temp_audio_{self.base_name}_{hash_str}.mp3"
        self.temp_vid = f"temp_video_{self.base_name}_{hash_str}.mp4"

        if self.debug_frames:
            self.debug_dir = f"debug_frames_{self.base_name}"
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[*] 디버그 모드 활성화: 프레임 이미지가 '{self.debug_dir}' 폴더에 저장됩니다.")

    def _extract_palette_independent(self, img_np):
        """이전 프레임의 기억을 배제하고 선택된 알고리즘으로 팔레트 독립 추출"""
        n_colors = 15
        
        if self.quant_algo == 'kmeans':
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_px = img_np[edges == 255]
            weighted = np.vstack([img_np.reshape(-1,3)] + [edge_px]*5) if len(edge_px)>0 else img_np.reshape(-1,3)
            
            unique_colors = len(np.unique(weighted, axis=0))
            if unique_colors < 1:
                raw = [(0,0,0)] * 15
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    km = KMeans(n_clusters=min(unique_colors, 15), init='k-means++', n_init=3, max_iter=30).fit(weighted)
                    raw = [tuple(c) for c in km.cluster_centers_]
        else:
            pil_img = Image.fromarray(img_np)
            method = Image.Quantize.MEDIANCUT if self.quant_algo == 'mediancut' else Image.Quantize.FASTOCTREE
            quantized = pil_img.quantize(colors=n_colors, method=method)
            pal = quantized.getpalette()
            
            raw = []
            if pal:
                pal = pal[:n_colors * 3]
                raw = [(pal[i], pal[i+1], pal[i+2]) for i in range(0, len(pal), 3)]

        final_pal_888 = raw
        while len(final_pal_888) < 15: final_pal_888.append((0,0,0))
        
        pal_333 = [tuple(int(round((c/255.0)*7)) for c in rgb) for rgb in final_pal_888[:15]]
        pal_888_np = np.zeros((16, 3), dtype=np.int32)
        for i, p in enumerate(pal_333):
            pal_888_np[i+1] = [int(c*255//7) for c in p]
            
        return pal_333, pal_888_np

    def run(self):
            print(f"[*] 공식 규격 완벽 인코딩 시작 (알고리즘: {self.quant_algo.upper()}, 디더: {self.dither_mode.upper()})")
            
            time_args = []
            if self.start_sec > 0: time_args.extend(["-ss", str(self.start_sec)])
            if self.end_sec: time_args.extend(["-to", str(self.end_sec)])

            # 1. MP3 오디오 추출
            subprocess.run(["ffmpeg", "-y"] + time_args + ["-i", self.input_video, "-vn", "-acodec", "libmp3lame", "-ac", "2", "-ar", "44100", "-b:a", "128k", "-id3v2_version", "0", self.temp_mp3], capture_output=True)

            # 2. 비디오 프리스케일링
            if not self.skip_prescale:
                print("[*] FFmpeg 512x384 사전 렌더링 중...")
                if self.aspect_mode == 'pad': vf_string = "scale=512:384:force_original_aspect_ratio=decrease:flags=lanczos,pad=512:384:-1:-1:color=black"
                elif self.aspect_mode == 'crop': vf_string = "scale=512:384:force_original_aspect_ratio=increase:flags=lanczos,crop=512:384"
                else: vf_string = "scale=512:384:flags=lanczos"

                subprocess.run(["ffmpeg", "-y"] + time_args + ["-i", self.input_video, "-an", "-vf", vf_string, "-r", "15", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "10", self.temp_vid], capture_output=True)
                cap = cv2.VideoCapture(self.temp_vid)
                orig_fps = 15.0
            else:
                cap = cv2.VideoCapture(self.input_video)
                orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            with open(self.temp_mp3, "rb") as f: mp3_data = f.read()
            out_f = open(self.output_mv2, "wb")
            
            # ==========================================================
            # 💡 [핵심 수정 1] 512B 폐기 -> 완벽한 16KB(16384) 공식 헤더 작성
            # ==========================================================
            official_header = bytearray(16384)
            official_header[0:8] = b'MMCSD_MV'          # 공통 시그니처
            official_header[8:16] = b'        '         # 8바이트 공백
            official_header[16:21] = b'v2.00'           # 버전 정보
            out_f.write(official_header)

            idx, mp3_off, bps = 0, 0, 16000 
            dither_flag = 2 if self.dither_mode == 'jjn' else (1 if self.dither_mode == 'fs' else 0)

            while cap.isOpened():
                if self.skip_prescale:
                    current_time = self.start_sec + (idx / 15.0)
                    if self.end_sec and current_time > self.end_sec: break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_time * orig_fps))

                ret, frame = cap.read()
                if not ret or mp3_off >= len(mp3_data): break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_512 = cv2.resize(img_rgb, (512, 384), interpolation=cv2.INTER_LANCZOS4) if self.skip_prescale else img_rgb
                
                pal_333, pal_888_np = self._extract_palette_independent(img_512)
                
                img_256 = cv2.resize(img_512, (256, 192), interpolation=cv2.INTER_AREA)
                img_rgb_diffused = _apply_dither_rgb(img_256, pal_888_np, dither_flag)
                
                pgt, ct = _encode_vram_optimal_search(img_rgb_diffused, pal_888_np)
                
                if self.debug_frames:
                    before_bgr = cv2.cvtColor(img_256, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(self.debug_dir, f"frame_{idx:04d}_before.png"), before_bgr)
                    
                    after_rgb = _reconstruct_msx_frame(pgt, ct, pal_888_np)
                    after_bgr = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(self.debug_dir, f"frame_{idx:04d}_after.png"), after_bgr)



                # ==========================================================
                # 💡 [원상 복구 1] 0번 색상 강제 주입 꼼수 폐기 -> 순수 15색 30바이트
                # ==========================================================
                pal_b = bytearray() # 기존의 bytearray([0, 0]) 삭제!
                for r, g, b in pal_333: 
                    pal_b.extend([(r<<4)|b, g])
                    
                # 혹시 모를 길이 오류 방지 (정확히 30바이트 고정)
                if len(pal_b) < 30: pal_b.extend(b'\x00' * (30 - len(pal_b)))
                elif len(pal_b) > 30: pal_b = pal_b[:30]

                # ==========================================================
                # 💡 [원상 복구 2] 32바이트 할당(12320) -> 공식 규격인 30바이트 할당(12318)
                # ==========================================================
                block = bytearray(b'\x55' * 16384) 
                block[0:6144] = pgt.tobytes()
                block[6144:12288] = ct.tobytes()
                block[12288:12318] = pal_b # 정확히 15색 (30바이트)
                
                # 오디오 청크 및 나머지 처리 (동일)
                target_a = int((idx + 1) * (bps / 15))
                sz = max(1, min(111, math.ceil((target_a - mp3_off) / 32))) 
                block[12800] = sz
                chunk = mp3_data[mp3_off : mp3_off + sz*32]
                block[12801 : 12801+len(chunk)] = chunk
                mp3_off += len(chunk)

                out_f.write(block)
                if idx % 10 == 0: sys.stdout.write(f"\r  > {idx} 프레임 정밀 최적화 인코딩 중..."); sys.stdout.flush()
                idx += 1

            print("\n")
            eof = bytearray(16384); eof[12318] = 0x01; eof[12800] = 0x22 
            out_f.write(eof); cap.release(); out_f.close()
            
            if os.path.exists(self.temp_mp3): os.remove(self.temp_mp3)
            if os.path.exists(self.temp_vid): os.remove(self.temp_vid)
            
            print(f"[!] 공식 규격(16KB 헤더) 완벽 인코딩 완료: {self.output_mv2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSX2 MV2 Perfect Frame Encoder (Math Optimal)")
    parser.add_argument("input", help="입력 동영상 파일 (.mp4)")
    parser.add_argument("output", help="출력 동영상 파일 (.mv2)")
    parser.add_argument("--algo", choices=['kmeans', 'mediancut', 'octree'], default='kmeans', help="팔레트 양자화 알고리즘")
    parser.add_argument("--dither", choices=['none', 'fs', 'jjn'], default='none', help="디더링 모드")
    parser.add_argument("-ss", dest="start", default=None)
    parser.add_argument("-to", dest="end", default=None)
    parser.add_argument("--aspect", choices=['pad', 'crop', 'force'], default='pad')
    parser.add_argument("--skip-prescale", action="store_true")
    parser.add_argument("--debug-frames", action="store_true", help="인코딩 전/후 프레임을 임시 폴더에 저장")
    
    args = parser.parse_args()
    
    MV2PerfectFrameEncoder(
        input_video=args.input, 
        output_mv2=args.output, 
        quant_algo=args.algo,
        dither_mode=args.dither, 
        start_time=args.start, 
        end_time=args.end, 
        aspect_mode=args.aspect, 
        skip_prescale=args.skip_prescale,
        debug_frames=args.debug_frames
    ).run()
