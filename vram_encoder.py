import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

class MSX2VRAMEncoder:
    def __init__(self):
        self.width = 256
        self.height = 192

    def _rgb888_to_rgb333(self, rgb):
        """24비트 트루컬러를 MSX2의 9비트 팔레트(R3 G3 B3) 포맷으로 다운샘플링"""
        r = int(round((rgb[0] / 255.0) * 7))
        g = int(round((rgb[1] / 255.0) * 7))
        b = int(round((rgb[2] / 255.0) * 7))
        return (r, g, b)

    def _rgb333_to_888(self, rgb333):
        """양자화된 333 컬러를 다시 888로 복원하여 거리 계산용으로 사용"""
        r, g, b = rgb333
        return (r * 255 // 7, g * 255 // 7, b * 255 // 7)

    def _color_distance(self, c1, c2):
        """두 RGB 색상 간의 유클리드 거리 제곱 (빠른 계산)"""
        return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2

    def generate_optimal_palette(self, image_array):
        """K-Means를 이용해 프레임별 최적의 15색 글로벌 팔레트 추출"""
        # 투명색(0번)을 제외한 15개 색상 추출
        pixels = image_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=15, n_init=3, random_state=42).fit(pixels)
        
        # 클러스터 중심점을 MSX2 규격(333)으로 변환 후 888로 재변환 (하드웨어 색상과 정확히 일치시키기 위함)
        msx_palette_333 = [self._rgb888_to_rgb333(center) for center in kmeans.cluster_centers_]
        
        # 1번부터 15번 팔레트까지 할당 (0번은 보통 투명/블랙으로 비워둠)
        palette = [(0,0,0)] * 16
        for i, p in enumerate(msx_palette_333):
            palette[i+1] = self._rgb333_to_888(p)
            
        return palette, msx_palette_333

    def encode_frame(self, image_path):
        """원본 이미지를 PGT(6144), CT(6144), Palette(30) 바이너리로 변환"""
        img = Image.open(image_path).convert('RGB').resize((self.width, self.height), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        
        # 1. 씬에 맞는 최적의 15색 팔레트 추출
        palette_888, palette_333 = self.generate_optimal_palette(img_array)
        
        pgt = bytearray(6144)
        ct = bytearray(6144)
        
        # 2. 8x1 블록(Color Clash) 인코딩 처리
        for y in range(self.height):
            for cx in range(32): # 가로 256픽셀을 8픽셀 단위로 나눔 (32개)
                x_start = cx * 8
                
                # 8x1 블록 픽셀 추출
                block_pixels = img_array[y, x_start:x_start+8]
                
                # 각 픽셀을 가장 가까운 팔레트 색상 인덱스로 매핑
                mapped_indices = []
                for px in block_pixels:
                    distances = [self._color_distance(px, p) for p in palette_888[1:16]]
                    best_idx = np.argmin(distances) + 1 # 1~15
                    mapped_indices.append(best_idx)
                
                # 블록 내에서 가장 많이 쓰인 2개의 색상 찾기 (Color Clash 강제 해결)
                unique, counts = np.unique(mapped_indices, return_counts=True)
                sorted_by_freq = unique[np.argsort(-counts)]
                
                fg_idx = sorted_by_freq[0]
                bg_idx = sorted_by_freq[1] if len(sorted_by_freq) > 1 else fg_idx
                
                # 3. 비트맵(PGT) 및 속성(CT) 데이터 생성
                p_byte = 0
                for i in range(8):
                    px_idx = mapped_indices[i]
                    # 현재 픽셀이 BG보다 FG에 더 가까우면 1(전경), 아니면 0(배경)
                    dist_fg = self._color_distance(palette_888[px_idx], palette_888[fg_idx])
                    dist_bg = self._color_distance(palette_888[px_idx], palette_888[bg_idx])
                    
                    if dist_fg <= dist_bg:
                        p_byte |= (1 << (7 - i))
                        
                c_byte = (fg_idx << 4) | bg_idx
                
                # MSX2 SCREEN 2 오프셋 공식 적용 (우리가 디먹서에서 썼던 공식을 완벽히 역산)
                b_idx = (y // 8) * 32 + cx
                off = (b_idx * 8) + (y % 8)
                
                pgt[off] = p_byte
                ct[off] = c_byte

        # 4. 30바이트 팔레트 바이너리 조립 (1번~15번 색상)
        pal_bytes = bytearray(30)
        for i in range(15):
            r, g, b = palette_333[i]
            # MSX2 팔레트 레지스터 포맷: Byte1(R, B), Byte2(G)
            pal_bytes[i*2] = (r << 4) | b
            pal_bytes[i*2 + 1] = g
            
        return pgt, ct, pal_bytes

# 테스트 실행
if __name__ == "__main__":
    encoder = MSX2VRAMEncoder()
    # PGT(6KB), CT(6KB), Palette(30B) 생성 테스트
    # pgt, ct, pal = encoder.encode_frame("test_frame.png")
    print("VRAM Encoder Ready.")
