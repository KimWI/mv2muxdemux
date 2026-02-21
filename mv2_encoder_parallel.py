import os, sys, math, subprocess, cv2, hashlib, warnings, argparse
import numpy as np
import concurrent.futures
from sklearn.cluster import KMeans
from numba import njit, prange

# ==========================================================
# 1. Numba JIT 고속 연산부 (GIL 완전 해제)
# ==========================================================
@njit(fastmath=True, cache=True, nogil=True)
def _get_dist_sq(c1, c2):
    return (int(c1[0]) - int(c2[0]))**2 + (int(c1[1]) - int(c2[1]))**2 + (int(c1[2]) - int(c2[2]))**2

@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def _apply_nearest(img_array, pal_888):
    """디더링 끄기 (None 모드)"""
    h, w = 192, 256
    indices = np.zeros((h, w), dtype=np.int32)
    for y in prange(h):
        for x in range(w):
            r, g, b = img_array[y, x, 0], img_array[y, x, 1], img_array[y, x, 2]
            min_d = 250000.0; best_i = 1
            for p_i in range(1, 16):
                d = (r - pal_888[p_i,0])**2 + (g - pal_888[p_i,1])**2 + (b - pal_888[p_i,2])**2
                if d < min_d: min_d, best_i = d, p_i
            indices[y, x] = best_i
    return indices

@njit(fastmath=True, cache=True, nogil=True)
def _apply_floyd_steinberg(img_array, pal_888):
    """Floyd-Steinberg 디더링 (FS 모드)"""
    h, w = 192, 256
    temp_img = img_array.astype(np.float32)
    indices = np.zeros((h, w), dtype=np.int32)
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
            
            indices[y, x] = best_i
            nr, ng, nb = pal_888[best_i, 0], pal_888[best_i, 1], pal_888[best_i, 2]
            er, eg, eb = (r - nr) * strength, (g - ng) * strength, (b - nb) * strength
            
            if x + 1 < w:
                temp_img[y, x+1, 0] += er * 0.4375; temp_img[y, x+1, 1] += eg * 0.4375; temp_img[y, x+1, 2] += eb * 0.4375
            if y + 1 < h:
                if x > 0:
                    temp_img[y+1, x-1, 0] += er * 0.1875; temp_img[y+1, x-1, 1] += eg * 0.1875; temp_img[y+1, x-1, 2] += eb * 0.1875
                temp_img[y+1, x, 0] += er * 0.3125; temp_img[y+1, x, 1] += eg * 0.3125; temp_img[y+1, x, 2] += eb * 0.3125
                if x + 1 < w:
                    temp_img[y+1, x+1, 0] += er * 0.0625; temp_img[y+1, x+1, 1] += eg * 0.0625; temp_img[y+1, x+1, 2] += eb * 0.0625
    return indices

@njit(fastmath=True, cache=True, nogil=True)
def _apply_jjn_dither(img_array, pal_888):
    """Jarvis-Judice-Ninke 디더링 (JJN 모드)"""
    h, w = 192, 256
    temp_img = img_array.astype(np.float32)
    indices = np.zeros((h, w), dtype=np.int32)
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
            
            indices[y, x] = best_i
            nr, ng, nb = pal_888[best_i, 0], pal_888[best_i, 1], pal_888[best_i, 2]
            er, eg, eb = (r - nr) * strength, (g - ng) * strength, (b - nb) * strength
            
            if x + 1 < w:
                temp_img[y, x+1, 0] += er * (7/48); temp_img[y, x+1, 1] += eg * (7/48); temp_img[y, x+1, 2] += eb * (7/48)
            if x + 2 < w:
                temp_img[y, x+2, 0] += er * (5/48); temp_img[y, x+2, 1] += eg * (5/48); temp_img[y, x+2, 2] += eb * (5/48)
                
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
                
    return indices

@njit(parallel=True, fastmath=True, cache=True, nogil=True)
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
# 2. 독립 병렬 작업 코어 (Worker Function)
# ==========================================================
def process_chunk(chunk_id, start_idx, frames_rgb, audio_info, mp3_data, dither_mode):
    blocks = bytearray()
    prev_centers = None
    
    for i, img_512 in enumerate(frames_rgb):
        idx = start_idx + i
        mp3_off, sz = audio_info[i]
        
        gray = cv2.cvtColor(img_512, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_px = img_512[edges == 255]
        weighted = np.vstack([img_512.reshape(-1,3)] + [edge_px]*5) if len(edge_px)>0 else img_512.reshape(-1,3)
        
        n_colors = min(len(np.unique(weighted, axis=0)), 15)
        if n_colors < 1:
            raw = [(0,0,0)] * 15
            prev_centers = None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                init_method = prev_centers if (prev_centers is not None and len(prev_centers) == n_colors) else 'k-means++'
                km = KMeans(n_clusters=n_colors, init=init_method, n_init=1, max_iter=15).fit(weighted)
                prev_centers = km.cluster_centers_
                raw = [tuple(c) for c in km.cluster_centers_]

        final_pal_888 = raw
        while len(final_pal_888) < 15: final_pal_888.append((0,0,0))
        
        pal_333 = [tuple(int(round((c/255.0)*7)) for c in rgb) for rgb in final_pal_888[:15]]
        pal_888_np = np.zeros((16, 3), dtype=np.int32)
        for j, p in enumerate(pal_333): pal_888_np[j+1] = [int(c*255//7) for c in p]
            
        img_256 = cv2.resize(img_512, (256, 192), interpolation=cv2.INTER_AREA)
        
        if dither_mode == 'jjn':
            indices_map = _apply_jjn_dither(img_256, pal_888_np)
        elif dither_mode == 'fs':
            indices_map = _apply_floyd_steinberg(img_256, pal_888_np)
        else:
            indices_map = _apply_nearest(img_256, pal_888_np)

        pgt, ct = _encode_vram_numba_master(pal_888_np, indices_map)
        pal_b = bytearray()
        for r, g, b in pal_333: pal_b.extend([(r<<4)|b, g])

        block = bytearray(b'\x55' * (15872 if idx == 0 else 16384))
        block[0:6144], block[6144:12288], block[12288:12318] = pgt.tobytes(), ct.tobytes(), pal_b
        
        block[12800] = sz
        chunk_audio = mp3_data[mp3_off : mp3_off + sz*32]
        block[12801 : 12801+len(chunk_audio)] = chunk_audio
        
        blocks.extend(block)
        
    return chunk_id, blocks

# ==========================================================
# 3. 메인 오케스트레이터 클래스
# ==========================================================
def parse_time_str(t_str):
    if not t_str: return 0.0
    try:
        parts = str(t_str).split(':')
        if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        elif len(parts) == 2: return float(parts[0])*60 + float(parts[1])
        return float(parts[0])
    except ValueError: return 0.0

class MV2ParallelMaster:
    def __init__(self, input_video, output_mv2, quant_algo='kmeans', dither_mode='none', start_time=None, end_time=None, aspect_mode='pad', skip_prescale=False):
        self.input_video = input_video
        self.output_mv2 = output_mv2
        self.dither_mode = dither_mode.lower()
        self.aspect_mode = aspect_mode.lower()
        self.skip_prescale = skip_prescale
        self.start_sec = parse_time_str(start_time)
        self.end_sec = parse_time_str(end_time) if end_time else None

        base = os.path.splitext(os.path.basename(input_video))[0]
        hash_str = hashlib.md5(f"{input_video}_{os.getpid()}".encode()).hexdigest()[:8]
        self.temp_mp3 = f"temp_audio_{base}_{hash_str}.mp3"
        self.temp_vid = f"temp_video_{base}_{hash_str}.mp4"

    def run(self):
        print(f"[*] 병렬 가속 인코딩 시작 (코어 수: {os.cpu_count()}, 디더: {self.dither_mode.upper()})")
        
        time_args = []
        if self.start_sec > 0: time_args.extend(["-ss", str(self.start_sec)])
        if self.end_sec: time_args.extend(["-to", str(self.end_sec)])

        subprocess.run(["ffmpeg", "-y"] + time_args + ["-i", self.input_video, "-vn", "-acodec", "libmp3lame", "-ac", "2", "-ar", "44100", "-b:a", "128k", "-id3v2_version", "0", self.temp_mp3], capture_output=True)

        if not self.skip_prescale:
            print("[*] FFmpeg 512x384 사전 렌더링 중...")
            if self.aspect_mode == 'pad': vf_string = "scale=512:384:force_original_aspect_ratio=decrease:flags=lanczos,pad=512:384:-1:-1:color=black"
            elif self.aspect_mode == 'crop': vf_string = "scale=512:384:force_original_aspect_ratio=increase:flags=lanczos,crop=512:384"
            else: vf_string = "scale=512:384:flags=lanczos"

            subprocess.run(["ffmpeg", "-y"] + time_args + ["-i", self.input_video, "-an", "-vf", vf_string, "-r", "15", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "10", self.temp_vid], capture_output=True)
            cap = cv2.VideoCapture(self.temp_vid)
            orig_fps = 15.0
        else:
            print("[!] FFmpeg 비디오 전처리 생략. OpenCV 직접 읽기 모드 작동.")
            cap = cv2.VideoCapture(self.input_video)
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        with open(self.temp_mp3, "rb") as f: mp3_data = f.read()
        out_f = open(self.output_mv2, "wb")
        out_f.write(bytearray(b'MV2 ').ljust(512, b'\x00')) 

        CHUNK_SIZE = 60 # 60프레임(4초) 단위 분할
        max_workers = os.cpu_count() or 4
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        futures = {}
        
        frames_buffer = []
        audio_info_buffer = []
        chunk_id = 0
        global_idx = 0
        mp3_off = 0
        bps = 16000
        next_chunk_to_write = 0
        
        print("[*] 프레임 분할 및 병렬 연산 돌입...")
        while cap.isOpened():
            if self.skip_prescale:
                current_time = self.start_sec + (global_idx / 15.0)
                if self.end_sec and current_time > self.end_sec: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_time * orig_fps))

            ret, frame = cap.read()
            if not ret or mp3_off >= len(mp3_data): break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_512 = cv2.resize(img_rgb, (512, 384), interpolation=cv2.INTER_LANCZOS4) if self.skip_prescale else img_rgb
            
            target_a = int((global_idx + 1) * (bps / 15))
            sz = max(1, min(111, math.ceil((target_a - mp3_off) / 32)))
            
            frames_buffer.append(img_512)
            audio_info_buffer.append((mp3_off, sz))
            mp3_off += sz * 32
            
            if len(frames_buffer) == CHUNK_SIZE:
                future = executor.submit(process_chunk, chunk_id, global_idx - CHUNK_SIZE + 1, frames_buffer, audio_info_buffer, mp3_data, self.dither_mode)
                futures[chunk_id] = future
                
                while len(futures) >= max_workers * 2:
                    if next_chunk_to_write in futures and futures[next_chunk_to_write].done():
                        _, blocks = futures.pop(next_chunk_to_write).result()
                        out_f.write(blocks)
                        print(f"  > 완료: 청크 {next_chunk_to_write} (프레임 {next_chunk_to_write*CHUNK_SIZE}~)")
                        next_chunk_to_write += 1
                    else:
                        break # 순서가 맞지 않으면 대기
                        
                frames_buffer = []; audio_info_buffer = []; chunk_id += 1
                
            global_idx += 1

        if frames_buffer:
            future = executor.submit(process_chunk, chunk_id, global_idx - len(frames_buffer), frames_buffer, audio_info_buffer, mp3_data, self.dither_mode)
            futures[chunk_id] = future
            
        while futures:
            if next_chunk_to_write in futures:
                _, blocks = futures.pop(next_chunk_to_write).result()
                out_f.write(blocks)
                print(f"  > 완료: 청크 {next_chunk_to_write} (마지막)")
                next_chunk_to_write += 1

        executor.shutdown()
        
        eof = bytearray(16384); eof[12318] = 0x01; eof[12800] = 0x22 
        out_f.write(eof); cap.release(); out_f.close()
        
        if os.path.exists(self.temp_mp3): os.remove(self.temp_mp3)
        if os.path.exists(self.temp_vid): os.remove(self.temp_vid)
        print(f"[!] 병렬 인코딩 완료: {self.output_mv2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSX2 MV2 Ultimate Parallel Encoder (v3.1)")
    parser.add_argument("input", help="입력 동영상 파일 (.mp4)")
    parser.add_argument("output", help="출력 동영상 파일 (.mv2)")
    parser.add_argument("--algo", choices=['kmeans', 'mediancut', 'octree'], default='mediancut')
    parser.add_argument("--dither", choices=['none', 'fs', 'jjn'], default='none', help="디더링 모드 (none: 끄기, fs: Floyd-Steinberg, jjn: Jarvis-Judice-Ninke)")
    parser.add_argument("-ss", dest="start", default=None, help="시작 시간 (예: 00:00:15)")
    parser.add_argument("-to", dest="end", default=None, help="종료 시간 (예: 00:00:30)")
    parser.add_argument("--aspect", choices=['pad', 'crop', 'force'], default='pad', help="화면 비율 유지 방식")
    parser.add_argument("--skip-prescale", action="store_true", help="FFmpeg 비디오 전처리 생략 (이미 가공된 영상일 경우)")
    
    args = parser.parse_args()
    
    MV2ParallelMaster(
        input_video=args.input, 
        output_mv2=args.output, 
        quant_algo=args.algo, 
        dither_mode=args.dither, 
        start_time=args.start, 
        end_time=args.end, 
        aspect_mode=args.aspect, 
        skip_prescale=args.skip_prescale
    ).run()
