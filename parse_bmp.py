import cv2
import os
import glob

# ==========================================
# 💡 [설정] 오리지널 BMP 화면 구조에 맞게 수정 필요
# ==========================================
BMP_DIR = "./eq_bmps"  # BMP 파일들이 모여있는 폴더
BAR_X_COORDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 9개 막대의 정중앙 X 좌표 (그림판으로 확인 필요!)
BASELINE_Y = 21      # 막대가 시작되는 바닥의 Y 좌표 (예: 192 높이일 경우 180 부근)
THRESHOLD = 200         # 배경(검정)과 막대(밝은색)를 구분할 밝기 기준
# ==========================================

def analyze_eq_bmps():
    bmp_files = sorted(glob.glob(os.path.join(BMP_DIR, "VIS*.BMP")))
    if not bmp_files:
        print(f"[!] {BMP_DIR} 폴더에 BMP 파일이 없습니다.")
        return

    print(f"[*] 총 {len(bmp_files)}개의 BMP 파일을 분석합니다...\n")
    
    results = []
    for idx, img_path in enumerate(bmp_files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        eq_values = []
        for x in BAR_X_COORDS:
            height = 0
            # 바닥(BASELINE_Y)부터 위로 올라가면서 픽셀이 밝은 동안 높이 증가
            for y in range(BASELINE_Y, 0, -1):
                if img[y, x] > THRESHOLD:
                    height += 1
                else:
                    break
            eq_values.append(height)
            
        results.append(eq_values)
        
        # 터미널에 실시간 출력
        print(f"Frame {idx:04d}: {eq_values}")

    # 최대 높이값 찾기 (오리지널 인코더가 0~15 스케일을 쓰는지, 0~31을 쓰는지 확인용)
    max_val = max(max(frame_eq) for frame_eq in results)
    print(f"\n[!] 분석 완료. 막대의 최대 픽셀 높이는 {max_val} 입니다.")
    print("이 데이터를 바탕으로 파이썬 FFT 밴드 공식을 튜닝할 수 있습니다.")

if __name__ == "__main__":
    analyze_eq_bmps()
