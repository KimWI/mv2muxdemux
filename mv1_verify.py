from PIL import Image

# 파일명은 실제 사용자님의 환경에 맞게 수정해 주십시오.
original_bmp = "원본_10번째_프레임.bmp"  # 방금 올려주려 하셨던 원본 파일
encoded_bmp = "frame_0010.bmp"         # 이전에 스크립트로 추출한 MV1 프레임

def count_unique_colors(image_path, label):
    try:
        img = Image.open(image_path).convert('RGB')
        # getcolors는 (빈도수, (R,G,B)) 형태의 리스트를 반환합니다.
        unique_colors = len(img.getcolors(maxcolors=16777216))
        print(f"[+] {label} 고유 색상 수: {unique_colors:,} 개")
        return unique_colors
    except Exception as e:
        print(f"[-] {label} 파일을 읽을 수 없습니다: {e}")
        return -1

print("=== 픽셀 정밀 타격: 색상 양자화(Quantization) 검증 ===")
count_orig = count_unique_colors(original_bmp, "원본 10번째 프레임")
count_enc = count_unique_colors(encoded_bmp, "MV1 인코딩 프레임 (frame_0010)")

if count_orig > 0 and count_enc > 0:
    print("\n[결론 도출]")
    if count_enc <= 16:
        print("=> 명백한 증거: 수많은 원본 색상이 MSX의 고정 16색 환경으로 강제 압축(디더링)되었습니다.")
    else:
        print("=> 가설 반박: 인코딩된 프레임이 16색을 초과합니다. (가변 팔레트가 쓰였을 가능성 존재)")
