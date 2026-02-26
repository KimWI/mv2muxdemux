import os, sys, math, subprocess, cv2, hashlib, warnings, argparse
import numpy as np
from sklearn.cluster import KMeans
from numba import njit, prange
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==========================================================
# 1. Numba JIT 고속 연산부
# ==========================================================
@njit(fastmath=True, cache=True)
def _get_dist_sq(c1, c2):
    return (int(c1[0]) - int(c2[0]))**2 + (int(c1[1]) - int(c2[1]))**2 + (int(c1[2]) - int(c2[2]))**2

@njit(fastmath=True, cache=True)
def _apply_dither_rgb(img_array, pal_888, mode):
    h, w = 192, 256
    temp_img = img_array.astype(np.float32)
    if mode == 0: return temp_img
        
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
                if x + 1 < w: temp_img[y, x+1, 0] += er * 0.4375; temp_img[y, x+1, 1] += eg * 0.4375; temp_img[y, x+1, 2] += eb * 0.4375
                if y + 1 < h:
                    if x > 0: temp_img[y+1, x-1, 0] += er * 0.1875; temp_img[y+1, x-1, 1] += eg * 0.1875; temp_img[y+1, x-1, 2] += eb * 0.1875
                    temp_img[y+1, x, 0] += er * 0.3125; temp_img[y+1, x, 1] += eg * 0.3125; temp_img[y+1, x, 2] += eb * 0.3125
                    if x + 1 < w: temp_img[y+1, x+1, 0] += er * 0.0625; temp_img[y+1, x+1, 1] += eg * 0.0625; temp_img[y+1, x+1, 2] += eb * 0.0625
            elif mode == 2: # JJN
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
def _apply_bayer_dither(img_array, spread=36.0):
    h, w = 192, 256
    bayer_4x4 = np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]], dtype=np.float32) / 16.0 - 0.5
    temp_img = np.zeros_like(img_array, dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            offset = bayer_4x4[y % 4, x % 4] * spread
            for c in range(3):
                temp_img[y, x, c] = min(max(img_array[y, x, c] + offset, 0.0), 255.0)
    return temp_img

@njit(parallel=True, fastmath=True, cache=True)
def _apply_bayer8_dither(img_array, spread=36.0):
    h, w = 192, 256
    bayer_8x8 = np.array([
        [0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],[12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],
        [3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],[15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]
    ], dtype=np.float32) / 64.0 - 0.5
    temp_img = np.zeros_like(img_array, dtype=np.float32)
    for y in prange(h):
        for x in range(w):
            offset = bayer_8x8[y % 8, x % 8] * spread
            for c in range(3):
                temp_img[y, x, c] = min(max(img_array[y, x, c] + offset, 0.0), 255.0)
    return temp_img

def _apply_bayer_dither_cuda(img_array, spread=30.0):
    h, w = 192, 256
    img_t = torch.tensor(img_array, dtype=torch.float32, device='cuda')
    bayer_4x4 = torch.tensor([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]], dtype=torch.float32, device='cuda') / 16.0 - 0.5
    bayer_map = bayer_4x4.repeat(h // 4, w // 4) * spread
    bayer_map = bayer_map.unsqueeze(2)
    out_t = torch.clamp(img_t + bayer_map, 0.0, 255.0)
    return out_t.cpu().numpy()

def _apply_bayer8_dither_cuda(img_array, spread=32.0):
    h, w = 192, 256
    img_t = torch.tensor(img_array, dtype=torch.float32, device='cuda')
    bayer_8x8 = torch.tensor([
        [0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],[12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],
        [3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],[15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]
    ], dtype=torch.float32, device='cuda') / 64.0 - 0.5
    bayer_map = bayer_8x8.repeat(h // 8, w // 8) * spread
    bayer_map = bayer_map.unsqueeze(2)
    out_t = torch.clamp(img_t + bayer_map, 0.0, 255.0)
    return out_t.cpu().numpy()

@njit(parallel=True, fastmath=True, cache=True)
def _encode_vram_optimal_search(img_rgb_float, pal_888):
    h, w = 192, 256
    pgt, ct = np.zeros(6144, dtype=np.uint8), np.zeros(6144, dtype=np.uint8)
    for y in prange(h):
        for cx in range(32):
            x_start = cx * 8
            block_rgb = img_rgb_float[y, x_start : x_start + 8]
            best_err = 1e12; best_fg = 1; best_bg = 1
            
            for i in range(1, 16):
                for j in range(1, i + 1):
                    err = 0.0
                    for p in range(8):
                        r, g, b = block_rgb[p]
                        r_cl = max(0.0, min(255.0, r)); g_cl = max(0.0, min(255.0, g)); b_cl = max(0.0, min(255.0, b))
                        d_i = (r_cl - pal_888[i,0])**2 + (g_cl - pal_888[i,1])**2 + (b_cl - pal_888[i,2])**2
                        d_j = (r_cl - pal_888[j,0])**2 + (g_cl - pal_888[j,1])**2 + (b_cl - pal_888[j,2])**2
                        err += d_i if d_i < d_j else d_j
                    if err < best_err:
                        best_err = err; best_fg = i; best_bg = j
            
            p_byte = 0
            for p in range(8):
                r, g, b = block_rgb[p]
                r_cl = max(0.0, min(255.0, r)); g_cl = max(0.0, min(255.0, g)); b_cl = max(0.0, min(255.0, b))
                d_fg = (r_cl - pal_888[best_fg,0])**2 + (g_cl - pal_888[best_fg,1])**2 + (b_cl - pal_888[best_fg,2])**2
                d_bg = (r_cl - pal_888[best_bg,0])**2 + (g_cl - pal_888[best_bg,1])**2 + (b_cl - pal_888[best_bg,2])**2
                if d_fg <= d_bg: p_byte |= (1 << (7 - p))
            
            off = ((y // 8) * 32 + cx) * 8 + (y % 8)
            pgt[off] = p_byte; ct[off] = (best_fg << 4) | best_bg
    return pgt, ct

def _encode_vram_optimal_search_cuda(img_rgb_float, pal_888):
    h, w = 192, 256
    
    # 텐서 복사 및 GPU 로드
    img_t = torch.tensor(img_rgb_float, dtype=torch.float32, device='cuda')
    img_t = torch.clamp(img_t, 0.0, 255.0)
    pal_t = torch.tensor(pal_888, dtype=torch.float32, device='cuda')
    
    # 텐서를 (블록 개수=192*32, 픽셀 8개, 채널 3개)로 재배열
    blocks = img_t.view(h, 32, 8, 3).reshape(-1, 8, 3)
    num_blocks = blocks.shape[0]

    # 각 픽셀과 팔레트 상의 15가지 색상(1번 인덱스부터) 사이의 제곱근 유클리드 거리 연산
    # blocks: [B, 8, 1, 3] / pal_t[1:]: [1, 1, 15, 3] -> diff: [B, 8, 15, 3]
    diff = blocks.unsqueeze(2) - pal_t[1:].unsqueeze(0).unsqueeze(0)
    dist = (diff ** 2).sum(dim=-1) # [B, 8, 15] 

    # 15개의 전경색(i)과 15개의 배경색(j) 간의 모든 조합 (총 225가지. i는 1~15, j는 1~i) 
    # 하지만 연산의 단순화를 위해 i, j 1~15 전체 매트릭스를 구성하고 GPU 브로드캐스팅 
    d_i = dist.unsqueeze(3) # [B, 8, 15, 1] - 전경색 거리 
    d_j = dist.unsqueeze(2) # [B, 8, 1, 15] - 배경색 거리
    
    # 픽셀마다 d_i 가 작은지 d_j 가 작은지 취합.
    min_dist = torch.minimum(d_i, d_j) # [B, 8, 15, 15]
    
    # 8픽셀 전체에 대한 에러 총합
    block_err = min_dist.sum(dim=1) # [B, 15, 15]
    
    # j <= i 조건 (j가 i보다 큰 부분은 무한대 처리하여 배제)
    mask = torch.tril(torch.ones(15, 15, dtype=torch.bool, device='cuda'))
    block_err = torch.where(mask, block_err, torch.tensor(float('inf'), device='cuda'))
    
    # 각 블록(B)에서 가장 에러가 적은 (fg, bg) 인덱스 도출
    flat_idx = block_err.view(num_blocks, -1).argmin(dim=1)
    best_i = flat_idx // 15
    best_j = flat_idx % 15
    
    # 각 조합에 따른 PGT 비트 계산
    best_di = dist[torch.arange(num_blocks), :, best_i] # [B, 8]
    best_dj = dist[torch.arange(num_blocks), :, best_j] # [B, 8]
    
    # d_i <= d_j 인 경우 비트 1로 설정 (전경색)
    bit_mask = (best_di <= best_dj).int() # [B, 8]
    
    # 8개의 비트를 하나의 1바이트로 압축
    shifts = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.int32, device='cuda')
    p_byte = (bit_mask << shifts.unsqueeze(0)).sum(dim=1).to(torch.uint8) # [B]
    
    # 실제 MSX 컬러코드 1-15는 인덱스 + 1
    fg = (best_i + 1).to(torch.uint8)
    bg = (best_j + 1).to(torch.uint8)
    c_byte = (fg << 4) | bg # [B]
    
    # GPU 텐서를 다시 CPU 평면 배열로 복사하여 정렬하기 
    # [192, 32] 를 MSX VRAM 메모리 주소 순서 (y//8 번째 줄의 x블록, y%8 스캔라인) 으로 매핑
    p_byte_cpu = p_byte.view(192, 32).cpu().numpy()
    c_byte_cpu = c_byte.view(192, 32).cpu().numpy()
    
    pgt = np.zeros(6144, dtype=np.uint8)
    ct = np.zeros(6144, dtype=np.uint8)
    
    for y in range(h):
        for cx in range(32):
            off = ((y // 8) * 32 + cx) * 8 + (y % 8)
            pgt[off] = p_byte_cpu[y, cx]
            ct[off] = c_byte_cpu[y, cx]
            
    return pgt, ct

@njit(fastmath=True, cache=True)
def _reconstruct_msx_frame(pgt, ct, pal_888_np):
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
    def __init__(self, input_video, output_mv2, quant_algo='kmeans', dither_mode='none', start_time=None, end_time=None, aspect_mode='pad', skip_prescale=False, use_temporal=False, debug_frames=False, scene_thresh=0.85, use_roi_face=False, use_roi_center=False, roi_center_spread=3.0, crop_up=0, crop_left=0, use_cuda=False):
        self.input_video = input_video
        self.output_mv2 = output_mv2
        self.quant_algo = quant_algo.lower()
        self.dither_mode = dither_mode.lower()
        self.aspect_mode = aspect_mode.lower()
        self.skip_prescale = skip_prescale
        self.start_sec = parse_time_str(start_time)
        self.end_sec = parse_time_str(end_time) if end_time else None
        
        self.use_temporal = use_temporal
        self.scene_thresh = scene_thresh  
        self.debug_frames = debug_frames 
        self.use_roi_face = use_roi_face
        self.use_roi_center = use_roi_center
        self.roi_center_spread = roi_center_spread
        self.crop_up = crop_up
        self.crop_left = crop_left
        self.use_cuda = use_cuda

        self.prev_hist = None
        self.prev_centroids = None
        
        # 💡 [핵심] OpenCV에 내장된 Haar Cascade 정면 얼굴 인식 모델 로드
        if self.use_roi_face:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("[!] 경고: OpenCV Haar Cascade 모델을 로드할 수 없습니다. ROI 기능이 무시됩니다.")
                self.use_roi_face = False

        self.base_name = os.path.splitext(os.path.basename(input_video))[0]
        hash_str = hashlib.md5(f"{input_video}_{os.getpid()}".encode()).hexdigest()[:8]
        self.temp_mp3 = f"temp_audio_{self.base_name}_{hash_str}.mp3"
        self.temp_vid = f"temp_video_{self.base_name}_{hash_str}.mp4"

        if self.debug_frames:
            self.debug_dir = f"debug_frames_{self.base_name}"
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[*] 디버그 모드 활성화: 프레임 이미지가 '{self.debug_dir}' 폴더에 저장됩니다.")

    def _detect_scene_change(self, img_np):
        hist = cv2.calcHist([img_np], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        
        is_scene_change = False
        if self.prev_hist is not None:
            score = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL)
            if score < self.scene_thresh:
                is_scene_change = True
        else:
            is_scene_change = True
            
        self.prev_hist = hist
        return is_scene_change

    def _extract_palette(self, img_np, is_scene_change):
        n_colors = 15
        
        if self.quant_algo == 'kmeans':
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # 💡 [핵심] 픽셀 복제(가중치)를 위한 마스크 배열 생성 (기본값: 1배)
            weight_mask = np.ones(gray.shape, dtype=np.uint8)
            
            # 1. 윤곽선 가중치 (기존 5배수 유지)
            edges = cv2.Canny(gray, 50, 150)
            weight_mask[edges == 255] = 5
            
            # 2. 얼굴 인식 ROI 가중치 (압도적인 30배수 할당!)
            face_detected = False
            if self.use_roi_face:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    # 얼굴 영역에 해당하는 마스크 배열 값을 30으로 덮어씌움
                    weight_mask[y:y+h, x:x+w] = 30
                    face_detected = True

            # 3. 중앙 집중 ROI 가중치 (가우시안 분포로 중앙일수록 높은 가중치)
            if self.use_roi_center:
                h, w = gray.shape
                # Meshgrid 
                y, x = np.ogrid[:h, :w]
                center_y, center_x = h / 2, w / 2
                
                # 정규화된 2D 가우시안 마스크 (중앙 1.0, 외곽 0.0)
                # 시그마 조정하여 집중도 변경 (사용자가 넘긴 스프레드 계수로 나누기)
                sigma_x, sigma_y = w / self.roi_center_spread, h / self.roi_center_spread
                gaussian_mask = np.exp(-(((x - center_x) ** 2) / (2 * sigma_x ** 2) + ((y - center_y) ** 2) / (2 * sigma_y ** 2)))
                
                # 강도 설정: 중앙은 최대 20배 가중치, 외곽은 기본값 + Alpha
                roi_center_weight = (gaussian_mask * 20).astype(np.uint8)
                
                # 기존 마스크(엣지나 얼굴)를 덮어쓰지 않고 가장 큰 가중치를 합산/초이스 
                weight_mask = np.maximum(weight_mask, roi_center_weight)

            # 4. 마스크(가중치)를 바탕으로 실제 픽셀 배열을 물리적으로 복제 (Numpy 매직)
            flat_img = img_np.reshape(-1, 3)
            flat_mask = weight_mask.reshape(-1)
            weighted_pixels = np.repeat(flat_img, flat_mask, axis=0)
            
            # 메모리 폭주(OOM) 방지: 픽셀이 너무 많아지면 30만 개로 샘플링 (K-Means 속도 유지)
            if len(weighted_pixels) > 300000:
                np.random.shuffle(weighted_pixels)
                weighted_pixels = weighted_pixels[:300000]
            
            unique_colors = len(np.unique(weighted_pixels, axis=0))
            if unique_colors < 1:
                raw = [(0,0,0)] * 15
                self.prev_centroids = None
            else:
                n_clusters = min(unique_colors, 15)
                
                if self.use_temporal and not is_scene_change and self.prev_centroids is not None and len(self.prev_centroids) == n_clusters:
                    init_val = np.array(self.prev_centroids)
                    n_init_val = 1 
                else:
                    init_val = 'k-means++'
                    n_init_val = 3 
                    
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    km = KMeans(n_clusters=n_clusters, init=init_val, n_init=n_init_val, max_iter=30).fit(weighted_pixels)
                    raw = [tuple(c) for c in km.cluster_centers_]
                    self.prev_centroids = raw.copy() 
                    
            return raw, face_detected # 💡 디버깅을 위해 얼굴 감지 여부 반환 변경

        else:
            pil_img = Image.fromarray(img_np)
            method = Image.Quantize.MEDIANCUT if self.quant_algo == 'mediancut' else Image.Quantize.FASTOCTREE
            quantized = pil_img.quantize(colors=n_colors, method=method)
            pal = quantized.getpalette()
            raw = [(pal[i], pal[i+1], pal[i+2]) for i in range(0, len(pal), 3)] if pal else []
            return raw, False

    def run(self):
        temporal_msg = f"활성화 (임계값: {self.scene_thresh})" if self.use_temporal else "비활성화"
        roi_msg = "얼굴 집중(ROI 30x)" if self.use_roi_face else "기본"
        print(f"[*] 공식 규격 인코딩 시작 (알고리즘: {self.quant_algo.upper()}, 디더: {self.dither_mode.upper()}, 시간적 일관성: {temporal_msg}, ROI: {roi_msg})")
        
        time_args = []
        if self.start_sec > 0: time_args.extend(["-ss", str(self.start_sec)])
        if self.end_sec: time_args.extend(["-to", str(self.end_sec)])

        subprocess.run(["ffmpeg", "-y"] + time_args + ["-i", self.input_video, "-vn", "-acodec", "libmp3lame", "-ac", "2", "-ar", "44100", "-b:a", "128k", "-id3v2_version", "0", self.temp_mp3], capture_output=True)

        if not self.skip_prescale:
            print("[*] FFmpeg 512x384 사전 렌더링 중...")
            
            # calculate pad/crop coordinates based on percent shift parameter (-100 to 100)
            # Default center formula: x=(ow-iw)/2, y=(oh-ih)/2
            # Modifier logic (e.g. crop_left=-100 pushes video full left, 100 pushes full right)
            x_shift = f"((ow-iw)/2)*(1.0+({self.crop_left}/100.0))"
            y_shift = f"((oh-ih)/2)*(1.0+({self.crop_up}/100.0))"
            cx_shift = f"((in_w-out_w)/2)*(1.0+({self.crop_left}/100.0))"
            cy_shift = f"((in_h-out_h)/2)*(1.0+({self.crop_up}/100.0))"

            if self.aspect_mode == 'pad': 
                vf_string = f"scale=512:384:force_original_aspect_ratio=decrease:flags=lanczos,pad=512:384:{x_shift}:{y_shift}:color=black"
            elif self.aspect_mode == 'crop': 
                vf_string = f"scale=512:384:force_original_aspect_ratio=increase:flags=lanczos,crop=512:384:{cx_shift}:{cy_shift}"
            else: 
                vf_string = "scale=512:384:flags=lanczos"
                
            input_args = ["-hwaccel", "cuda", "-i", self.input_video] if self.use_cuda else ["-i", self.input_video]
            codec_args = ["-c:v", "h264_nvenc", "-preset", "p1"] if self.use_cuda else ["-c:v", "libx264", "-preset", "ultrafast"]
            
            subprocess.run(["ffmpeg", "-y"] + time_args + input_args + ["-an", "-vf", vf_string, "-r", "15"] + codec_args + ["-crf", "10", self.temp_vid], capture_output=True)
            cap = cv2.VideoCapture(self.temp_vid)
            orig_fps = 15.0
        else:
            cap = cv2.VideoCapture(self.input_video)
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        with open(self.temp_mp3, "rb") as f: mp3_data = f.read()
        out_f = open(self.output_mv2, "wb")
        
        official_header = bytearray(16384)
        official_header[0:8] = b'MMCSD_MV'
        official_header[8:16] = b'        '
        official_header[16:21] = b'v2.00'
        out_f.write(official_header)

        idx, mp3_off, bps = 0, 0, 16000 
        
        while cap.isOpened():
            if self.skip_prescale:
                current_time = self.start_sec + (idx / 15.0)
                if self.end_sec and current_time > self.end_sec: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_time * orig_fps))

            ret, frame = cap.read()
            if not ret or mp3_off >= len(mp3_data): break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_512 = cv2.resize(img_rgb, (512, 384), interpolation=cv2.INTER_LANCZOS4) if self.skip_prescale else img_rgb
            
            is_scene_change = self._detect_scene_change(img_512)
            
            # 💡 [수정] 언패킹 로직 변경 및 팔레트 정렬 적용
            raw_pal, face_detected = self._extract_palette(img_512, is_scene_change)
            
            final_pal_888 = raw_pal
            while len(final_pal_888) < 15: final_pal_888.append((0,0,0))
            final_pal_888 = final_pal_888[:15]
            final_pal_888.sort(key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
            
            pal_333 = [tuple(int(round((c/255.0)*7)) for c in rgb) for rgb in final_pal_888]
            pal_888_np = np.zeros((16, 3), dtype=np.int32)
            for i, p in enumerate(pal_333):
                pal_888_np[i+1] = [int(c*255//7) for c in p]
            
            img_256 = cv2.resize(img_512, (256, 192), interpolation=cv2.INTER_AREA)
            
            if self.dither_mode == 'bayer':
                img_rgb_diffused = _apply_bayer_dither(img_256.astype(np.float32))
            elif self.dither_mode == 'bayer8':
                img_rgb_diffused = _apply_bayer8_dither(img_256.astype(np.float32))
            else:
                dither_flag = 2 if self.dither_mode == 'jjn' else (1 if self.dither_mode == 'fs' else 0)
                img_rgb_diffused = _apply_dither_rgb(img_256, pal_888_np, dither_flag)
            
            # 💡 [핵심] GPU 병렬 처리가 활성화 되어있는지 검사 후 PyTorch 분기, 그렇지 않으면 기존 Numba CPU로 분기
            if self.use_cuda and HAS_TORCH and torch.cuda.is_available():
                pgt, ct = _encode_vram_optimal_search_cuda(img_rgb_diffused, pal_888_np)
            else:
                pgt, ct = _encode_vram_optimal_search(img_rgb_diffused, pal_888_np)

            if self.debug_frames:
                before_bgr = cv2.cvtColor(img_256, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.debug_dir, f"frame_{idx:04d}_before.png"), before_bgr)
                
                after_rgb = _reconstruct_msx_frame(pgt, ct, pal_888_np)
                after_bgr = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.debug_dir, f"frame_{idx:04d}_after.png"), after_bgr)
            
            pal_b = bytearray() 
            for r, g, b in pal_333: pal_b.extend([(r<<4)|b, g])
            if len(pal_b) < 30: pal_b.extend(b'\x00' * (30 - len(pal_b)))
            elif len(pal_b) > 30: pal_b = pal_b[:30]

            block = bytearray(b'\x55' * 16384) 
            block[0:6144] = pgt.tobytes()
            block[6144:12288] = ct.tobytes()
            block[12288:12318] = pal_b
            
            target_a = int((idx + 1) * (bps / 15))
            sz = max(1, min(111, math.ceil((target_a - mp3_off) / 32))) 
            block[12800] = sz
            chunk = mp3_data[mp3_off : mp3_off + sz*32]
            block[12801 : 12801+len(chunk)] = chunk
            mp3_off += len(chunk)
            
            out_f.write(block)
            
            status_char = "✂️ 씬 전환!" if is_scene_change else ("👤 얼굴 집중!" if face_detected else "  ")
            sys.stdout.write(f"\r  > {idx} 프레임 인코딩 중... {status_char}        ")
            sys.stdout.flush()
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
    parser.add_argument("--dither", choices=['none', 'fs', 'jjn', 'bayer', 'bayer8'], default='none', help="디더링 모드")
    parser.add_argument("--temporal", action="store_true", help="[추천] 씬 감지를 포함한 팔레트 시간적 일관성(깜빡임 방지) 활성화")
    parser.add_argument("--scene-thresh", type=float, default=0.85, help="씬 전환 감지 임계값 (기본: 0.85 / 예민하게: 0.93)")
    
    # 💡 [추가] 얼굴 인식 및 화면 중앙 집중 패턴 (K-Means 전용)
    parser.add_argument("--roi-face", action="store_true", help="인물/캐릭터 얼굴에 팔레트 색상을 대거 할당 (KMeans 전용)")
    parser.add_argument("--roi-center", action="store_true", help="화면 중앙부에 팔레트 색상을 집중 할당하는 2D 가우시안 ROI 패턴 적용 (KMeans 전용)")
    parser.add_argument("--roi-center-spread", type=float, default=3.0, help="중앙 ROI 퍼짐 정도 (작을수록 화면 전체로 균등. 기본: 3.0)")
    parser.add_argument("--cuda", action="store_true", help="NVIDIA CUDA(NVENC/NVDEC)를 사용하여 FFmpeg 다운스케일 렌더링을 매우 가속화합니다.")
    
    parser.add_argument("-ss", dest="start", default=None)
    parser.add_argument("-to", dest="end", default=None)
    parser.add_argument("--aspect", choices=['pad', 'crop', 'force'], default='pad')
    parser.add_argument("--skip-prescale", action="store_true")
    
    # 💡 [추가] 여백 패스 및 크롭 위치 조절 (-100 ~ 100 퍼센트 배열 스크롤)
    parser.add_argument("--crop-up", type=float, default=0, help="비디오 종횡비 패딩시 상하 강제 이동 퍼센트 (-100:상단 딱붙 ~ 100:하단 딱붙)")
    parser.add_argument("--crop-left", type=float, default=0, help="비디오 종횡비 패딩시 좌우 강제 이동 퍼센트 (-100:좌측 딱붙 ~ 100:우측 딱붙)")
    
    parser.add_argument("--debug-frame", "--debug-frames", dest="debug_frames", action="store_true", help="인코딩 전/후 프레임을 임시 폴더에 저장")
    
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
        use_temporal=args.temporal,
        debug_frames=args.debug_frames,
        scene_thresh=args.scene_thresh,
        use_roi_face=args.roi_face,
        use_roi_center=args.roi_center,
        roi_center_spread=args.roi_center_spread,
        crop_up=args.crop_up,
        crop_left=args.crop_left,
        use_cuda=args.cuda
        ).run()
