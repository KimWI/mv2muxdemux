import os
import glob
from PIL import Image

def get_luminance(rgb):
    """RGB 튜플을 입력받아 밝기(Luminance) 값을 반환합니다."""
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b

def extract_number(filename):
    """파일명에서 숫자를 추출하여 정렬 기준으로 사용합니다."""
    base = os.path.basename(filename)
    numbers = ''.join(filter(str.isdigit, base))
    return int(numbers) if numbers else 0

def analyze_pixel_colors(directory_path):
    search_pattern = os.path.join(directory_path, '*.BMP')
    bmp_files = sorted(glob.glob(search_pattern), key=extract_number)
    
    total_frames = len(bmp_files)
    if total_frames < 2:
        print("검증을 위해서는 최소 2개 이상의 BMP 프레임이 필요합니다.")
        return

    print(f"총 {total_frames}개의 프레임에서 실제 픽셀 컬러를 추출하여 분석합니다...\n")

    changed_color_set_count = 0
    identical_color_set_count = 0
    over_color_limit_count = 0
    previous_sorted_colors = None

    for i, file_path in enumerate(bmp_files):
        try:
            with Image.open(file_path) as img:
                # 인덱스 모드든 24비트 모드든 동일한 픽셀 비교를 위해 RGB 모드로 강제 변환
                img_rgb = img.convert('RGB')
                
                # 이미지 내에 사용된 모든 고유 색상 추출 
                # 반환 형태: [(픽셀 수, (R, G, B)), ...]
                # 16색 환경을 가정하므로 넉넉하게 maxcolors=256으로 설정
                colors_in_image = img_rgb.getcolors(maxcolors=256)
                
                if colors_in_image is None:
                    # 256색 이상이 사용된 경우 (디더링 과정에서 True Color로 섞였거나 16색 제한이 풀린 경우)
                    over_color_limit_count += 1
                    continue
                
                # 픽셀 빈도수는 무시하고 순수 RGB 색상 튜플만 추출
                unique_colors = [color_info[1] for color_info in colors_in_image]
                
                # 추출된 색상을 밝기(Luminance) 순으로 정렬
                sorted_colors = sorted(unique_colors, key=get_luminance)
                
                if previous_sorted_colors is not None:
                    # 정렬된 색상 리스트(컬러셋)를 이전 프레임과 픽셀 단위로 완벽히 대조
                    if sorted_colors != previous_sorted_colors:
                        changed_color_set_count += 1
                    else:
                        identical_color_set_count += 1
                        
                previous_sorted_colors = sorted_colors
                
        except Exception as e:
            print(f"파일을 읽는 중 오류 발생 ({file_path}): {e}")

    # 분석 결과 및 통계 출력
    valid_comparisons = total_frames - 1 - over_color_limit_count
    
    print("-" * 50)
    print("[ 실제 픽셀 컬러 변동 통계 분석 결과 ]")
    print("-" * 50)
    print(f"전체 파일 수      : {total_frames} 개")
    print(f"16색(256색 이하) 초과 프레임 : {over_color_limit_count} 개 (분석 제외)")
    print(f"총 비교 횟수      : {valid_comparisons} 회")
    print(f"컬러셋 변경 횟수  : {changed_color_set_count} 회")
    print(f"컬러셋 유지 횟수  : {identical_color_set_count} 회")
    
    if valid_comparisons > 0:
        change_rate = (changed_color_set_count / valid_comparisons) * 100
        print(f"\n=> 프레임 간 컬러셋(16색) 변경률: {change_rate:.2f}%")
        
        if change_rate > 90:
            print("\n[기술적 결론] 프레임마다 실제 화면에 찍힌 픽셀의 색상 구성(RGB 값)이 계속 달라집니다.")
            print("이는 하나의 고정된 16색 팔레트를 사용한 것이 아니라, 매 프레임 원본 영상의 색감에 맞춰 최적의 16색을 동적으로 재계산(Color Quantization)하여 디더링했다는 확실한 증거입니다.")
        elif change_rate == 0:
            print("\n[기술적 결론] 모든 프레임이 정확히 동일한 색상 조합을 사용하고 있습니다. 동적 갱신이 아닌 고정 팔레트 방식입니다.")

#analyze_palette_changes('/run/media/muhanpong/Elements/Documents/Shared/MMCSD_AVGEN_v1.09/mmcsd_avgen.tmp')
analyze_pixel_colors('/run/media/muhanpong/Elements/Documents/Shared/MMCSD_AVGEN_v1.09/mmcsd_avgen.tmp')
 
