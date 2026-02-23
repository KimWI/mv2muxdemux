import os, argparse

def verify_mv1_spec(file_path):
    block_size = 16384
    palette_offset = 12288 # 제시해주신 12288번지 (0x3000)
    
    with open(file_path, 'rb') as f:
        for i in range(3): # 처음 3개 블록만 확인
            f.seek(i * block_size + palette_offset)
            palette_data = f.read(32)
            
            print(f"--- Block {i} Palette Area (at Offset {palette_offset}) ---")
            hex_dump = ' '.join(f"{b:02X}" for b in palette_data)
            print(hex_dump)
            
            # MSX2 팔레트 특징: 2바이트 중 두 번째 바이트는 항상 0x0~0x7 사이여야 함
            is_valid_pattern = all((palette_data[j] & 0xF0 == 0) for j in range(1, 32, 2))
            if is_valid_pattern:
                print("결과: MSX2 팔레트 데이터 패턴과 일치합니다!")
            else:
                print("결과: 팔레트 패턴이 아닙니다. MV1은 오프셋이 다를 수 있습니다.")
            print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSX2 MV / MV2 ct tester")
    parser.add_argument("file_path", help="입력 MV, MV2(.MV, MV2)")
    args = parser.parse_args()
    verify_mv1_spec(args.file_path)

