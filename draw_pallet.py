import os
import sys
import glob
from PIL import Image, ImageDraw
from collections import Counter
import colorsys

def get_sorted_palette(image_path, num_colors=32):
    """이미지에서 상위 16색을 추출하고 시각적 연속성을 위해 정렬합니다."""
    img = Image.open(image_path).convert('RGB')
    pixels = list(img.getdata())
    counts = Counter(pixels)
    
    # 상위 16색 추출
    most_common = [color for color, count in counts.most_common(num_colors)]
    
    # 16색이 안 될 경우 검은색(0,0,0)으로 패딩
    while len(most_common) < num_colors:
        most_common.append((0, 0, 0))
        
    # 인접 프레임 간 동일한 계열의 색상이 같은 열(Column)에 위치하도록 HSV 기준으로 정렬
    #most_common.sort(key=lambda rgb: colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255))
    return most_common

def draw_palette_timeline(directory, pattern):
    # 1. 파일 목록 로드 및 프레임 순서 정렬
    pic_files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not pic_files:
        print("BMP 파일을 찾을 수 없습니다.")
        return

    num_frames = len(pic_files)
    
    # 2. 캔버스 규격 계산
    # 가로: 8 픽셀 * 16 컬러 = 128 픽셀
    # 세로: 2 픽셀 * 프레임 수
    canvas_width = 8 * 16
    canvas_height = num_frames * 2
    
    print(f"총 {num_frames}개의 프레임을 분석하여 {canvas_width} x {canvas_height} 픽셀의 이미지를 생성합니다.")

    # 3. 빈 캔버스 생성
    out_img = Image.new('RGB', (canvas_width, canvas_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(out_img)

    # 4. 프레임별 라인 드로잉
    for i, file_path in enumerate(pic_files):
        palette = get_sorted_palette(file_path, 16)
        
        # 현재 프레임이 그려질 Y 좌표 (세로 2px 단위)
        y_start = i * 2
        y_end = y_start + 2
        
        # 16개 색상을 가로 8px 단위로 그리기
        for j, color in enumerate(palette):
            x_start = j * 8
            x_end = x_start + 8
            
            # 사각형 그리기 (x0, y0, x1, y1)
            draw.rectangle([x_start, y_start, x_end, y_end], fill=color)

    # 5. 결과물 저장
    output_filename = "palette_timeline.png"
    out_img.save(output_filename)
    print(f"[완료] '{output_filename}' 파일이 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    draw_palette_timeline(sys.argv[1], sys.argv[2])
