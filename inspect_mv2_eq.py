import sys

def analyze_mv2_eq(filepath, num_frames_to_show=3000):
    try:
        with open(filepath, "rb") as f:
            # 1. 글로벌 헤더 (16KB) 읽기
            header = f.read(16384)
            if len(header) < 16384:
                print("[!] 파일이 너무 짧습니다.")
                return
                
            sig = header[0:8].decode('ascii', errors='ignore')
            version = header[16:21].decode('ascii', errors='ignore')
            print(f"[*] 분석 대상 파일: {filepath}")
            print(f"[*] 시그니처: [{sig}], 버전: [{version}]\n")
            
            frame_idx = 0
            max_val = 0
            min_val = 255
            
            print(f"[*] 오프셋 12320 ~ 12329 (10 Bytes) 원시 데이터 추출 중...")
            print("-" * 50)
            
            while True:
                block = f.read(16384)
                if len(block) < 16384:
                    break # 파일 끝
                    
                # EOF 프레임 체크 (12318 오프셋이 0x01이면 마지막 더미 프레임)
                if block[12318] == 0x01:
                    break
                    
                # 💡 [핵심] 12320부터 10바이트 추출
                eq_data = block[12320:12330] 
                eq_values = list(eq_data)
                
                # 통계용 최대/최소값 갱신
                max_val = max(max_val, max(eq_values))
                min_val = min(min_val, min(eq_values))
                
                # 처음 num_frames_to_show 개수만큼만 터미널에 출력
                if frame_idx < num_frames_to_show:
                    print(f"Frame {frame_idx:04d}: {eq_values}")
                elif frame_idx == num_frames_to_show:
                    print(f"... (이후 프레임 출력 생략) ...")
                    
                frame_idx += 1
                
            print("-" * 50)
            print(f"[!] 총 {frame_idx} 프레임 분석 완료.")
            print(f"[!] 원시 데이터(Raw Byte) 값의 범위: 최소 {min_val} ~ 최대 {max_val}")
            
            # 💡 교차 검증 결과 판독
            print("\n[🔍 판독 결과]")
            if max_val == 21 or max_val == 22:
                print(">> BMP 픽셀 높이와 정확히 일치합니다! 오리지널 인코더는 0~21 스케일을 파일에 그대로 기록했습니다.")
            elif max_val == 15:
                print(">> 원본 바이너리는 4비트(0~15) 스케일이 맞습니다! BMP 이미지는 렌더링 시에만 확대된 것입니다.")
            elif max_val > 22:
                print(f">> 예상 밖입니다. 1바이트 스케일(0~255) 중 최대 {max_val}까지 사용하고 있습니다.")
                
    except FileNotFoundError:
        print(f"[!] 파일을 찾을 수 없습니다: {filepath}")
    except Exception as e:
        print(f"[!] 에러 발생: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_mv2_eq(sys.argv[1])
    else:
        print("사용법: python inspect_mv2_eq.py original_file.mv2")
