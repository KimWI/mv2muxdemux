import sys
import os

def check_pcm_tail_palette(file_path, block_size=16384):
    """
    파일의 각 16KB 블록 말미 32바이트를 추출하여 
    MSX2 팔레트 데이터 패턴이 있는지 분석합니다.
    """
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        return

    file_size = os.path.getsize(file_path)
    total_blocks = file_size // block_size
    
    print(f"파일 분석 시작: {file_path}")
    print(f"전체 크기: {file_size} Bytes (약 {total_blocks}개 블록)")
    print("-" * 60)

    tail_offset = block_size - 32 # 16KB 블록의 마지막 32바이트 지점
    tail_offset = 12288+944+8 # PGT+CT+PCM 이후 32바이트 지점
    
    with open(file_path, 'rb') as f:
        # 처음 5개 블록과 마지막 1개 블록 정도를 확인 (필요시 range 조절 가능)
        blocks_to_check = list(range(min(500, total_blocks)))
        
        for i in blocks_to_check:
            f.seek((i * block_size) + tail_offset)
            tail_data = f.read(32)
            
            if len(tail_data) < 32:
                break
                
            print(f"--- Block {i} Tail (Offset: {i*block_size + tail_offset:#08X}) ---")
            
            # 16진수 덤프 출력
            hex_dump = ' '.join(f"{b:02X}" for b in tail_data)
            print(f"HEX: {hex_dump}")
            
            # MSX2 팔레트 패턴 검사: 2바이트 단위로 두 번째 바이트(Green)의 상위 비트가 0인지 확인
            # (앞서 확인한 55 55 같은 특수 마커를 고려하여 30바이트만 우선 검사)
            is_valid = all((tail_data[j] & 0xF0 == 0) for j in range(1, 31, 2))
            
            if is_valid:
                print("결과: MSX2 팔레트 데이터 패턴 검출 가능성 높음")
            else:
                print("결과: 표준 팔레트 규격과 일치하지 않음")
            print("-" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python 스크립트이름.py <파일명>")
        print("예시: python analyze_mv.py 1MB.MV")
    else:
        target_file = sys.argv[1]
        check_pcm_tail_palette(target_file)
