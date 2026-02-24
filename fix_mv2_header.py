import sys
import os

def fix_mv2_header(input_file, output_file):
    print(f"[*] 분석 시작: {input_file}")
    
    with open(input_file, 'rb') as f:
        data = f.read()

    if len(data) < 16384:
        print("[X] 파일 크기가 너무 작습니다. 정상적인 MV 파일이 아닙니다.")
        return

    sig = data[0:8]
    
    # 케이스 1: 우리가 작성했던 커스텀 인코더 파일 (512B 헤더)
    if sig.startswith(b'MV2 '):
        print("[!] 커스텀 인코더(512B 헤더)로 작성된 파일을 감지했습니다.")
        print("[*] 공식 16KB 블록 헤더 규격으로 마이그레이션을 시작합니다...")
        
        # 1. 새로운 16KB(16384바이트) 공식 글로벌 헤더 생성
        official_header = bytearray(16384)
        official_header[0:8] = b'MMCSD_MV'          # 시그니처
        official_header[8:16] = b'        '         # 8바이트 공백
        official_header[16:21] = b'v2.00'           # 버전 정보
        
        # 2. 첫 번째 프레임(Frame 0) 확장 (15872 -> 16384 바이트)
        # 기존 커스텀 파일은 헤더가 512바이트였으므로, Frame 0이 512~16383에 위치함
        frame0 = data[512 : 512 + 15872]
        padded_frame0 = bytearray(frame0)
        padded_frame0.extend(b'\x55' * 512) # 16KB 규격을 맞추기 위해 끝에 0x55 패딩
        
        # 3. 나머지 프레임들 (Frame 1 ~ EOF)
        rest_frames = data[512 + 15872:]
        
        # 4. 바이너리 조합 및 저장
        with open(output_file, 'wb') as f_out:
            f_out.write(official_header)
            f_out.write(padded_frame0)
            f_out.write(rest_frames)
            
        print(f"[+] 마이그레이션 완료! (출력: {output_file})")
        print("    └> 헤더 16KB 확장 및 첫 프레임 16KB 정렬이 완벽하게 교정되었습니다.")
        
    # 케이스 2: 공식 인코더 규격이지만 버전 문자열이 손상된 파일
    elif sig == b'MMCSD_MV':
        print("[*] 이 파일은 이미 16KB 공식 헤더(MMCSD_MV)를 사용 중입니다.")
        version = data[16:21]
        
        if version != b'v2.00':
            print(f"[!] 버전 정보가 [{version}]로 잘못 기재되어 있습니다. v2.00으로 패치합니다.")
            modified_data = bytearray(data)
            modified_data[16:21] = b'v2.00'
            
            with open(output_file, 'wb') as f_out:
                f_out.write(modified_data)
            print(f"[+] 버전 헤더 복구 완료! (출력: {output_file})")
        else:
            print("[+] 헤더와 버전이 모두 정상입니다. 수정할 필요가 없습니다.")
            
    else:
        print(f"[X] 알 수 없는 파일 시그니처입니다: {sig}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("==================================================")
        print(" MV2 공식 헤더 교정 및 복구 툴")
        print("==================================================")
        print("사용법: python fix_mv2_header.py <원본파일.mv2> <출력파일.mv2>")
        print("예  시: python fix_mv2_header.py custom.mv2 official_fixed.mv2")
    else:
        fix_mv2_header(sys.argv[1], sys.argv[2])
