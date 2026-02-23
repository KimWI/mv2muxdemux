import os
import sys
import argparse
import subprocess
import numpy as np

def reconstruct_screen2_frame(pgt, ct, palette):
    """
    MSX2 SCREEN 2 VRAM 구조를 RGB888 평면 비트맵으로 복원합니다.
    """
    # 가로 256, 세로 192, 3채널(RGB)
    frame_rgb = np.zeros((192, 256, 3), dtype=np.uint8)
    
    for y in range(192):
        for cx in range(32):
            # 타일 기반 인덱싱: (y//8)*32 + cx 는 타일 번호, *8 + y%8 은 타일 내 라인
            tile_idx = ((y // 8) * 32 + cx) * 8 + (y % 8)
            p_byte = pgt[tile_idx]
            c_byte = ct[tile_idx]
            
            fg_idx = c_byte >> 4
            bg_idx = c_byte & 0x0F

            for bit in range(8):
                is_fg = (p_byte & (1 << (7 - bit))) != 0
                color = palette[fg_idx if is_fg else bg_idx]
                # x 좌표는 cx * 8 + bit
                frame_rgb[y, cx * 8 + bit] = color
    return frame_rgb

def transcode_mv1(input_path, output_path):
    BLOCK_SIZE = 16384
    FPS = 12
    SAMPLE_RATE = 11520 # 960 bytes * 12 fps

    if not os.path.exists(input_path):
        print(f"[!] 파일을 찾을 수 없습니다: {input_path}")
        return

    file_size = os.path.getsize(input_path)
    num_frames = file_size // BLOCK_SIZE
    
    # 임시 파일 생성 (프로세스 ID 포함으로 충돌 방지)
    temp_vid = f"temp_vid_{os.getpid()}.raw"
    temp_aud = f"temp_aud_{os.getpid()}.pcm"

    print(f"[*] 분석 시작: {num_frames} 프레임 감지")
    print(f"[*] 규격 확인: {FPS}fps / {SAMPLE_RATE}Hz 8-bit PCM")

    try:
        with open(input_path, "rb") as f, \
             open(temp_vid, "wb") as f_v, \
             open(temp_aud, "wb") as f_a:

            for idx in range(num_frames):
                block = f.read(BLOCK_SIZE)
                if len(block) < BLOCK_SIZE: break

                # 오프셋 정밀 슬라이싱
                pgt = block[0x0000:0x1800]
                ct  = block[0x1800:0x3000]
                pcm = block[0x3000:0x33C0]
                pal = block[0x33C0:0x33E0]

                # 오디오 스트림 누적
                f_a.write(pcm)

                # 팔레트 변환 (RGB333 to RGB888)
                palette = []
                for i in range(16):
                    b1, b2 = pal[i*2], pal[i*2+1]
                    r = ((b1 >> 4) & 0x07) * 255 // 7
                    b = (b1 & 0x07) * 255 // 7
                    g = (b2 & 0x07) * 255 // 7
                    palette.append((r, g, b))

                # 비디오 프레임 생성
                frame = reconstruct_screen2_frame(pgt, ct, palette)
                f_v.write(frame.tobytes())

                if idx % 50 == 0:
                    sys.stdout.write(f"\r  > 처리 중: {idx}/{num_frames}")
                    sys.stdout.flush()

        print("\n[*] FFmpeg Muxing 시작...")
        # libx264(비디오) + aac(오디오) 결합
        cmd = [
            "ffmpeg", "-y",
            "-f", "u8", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", temp_aud,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x192", "-r", str(FPS), "-i", temp_vid,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "18",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
        
        # 상세 에러 확인을 위해 capture_output 사용 가능
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[!] FFmpeg 에러:\n{result.stderr}")
        else:
            print(f"[!] 변환 성공: {output_path}")

    finally:
        # 종료 전 임시 파일 삭제
        for t in [temp_vid, temp_aud]:
            if os.path.exists(t): os.remove(t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MV1 to MP4 Transcoder (12fps/11.5kHz)")
    parser.add_argument("input", help="입력 MV1 파일")
    args = parser.parse_args()
    
    output_name = os.path.splitext(args.input)[0] + ".mp4"
    transcode_mv1(args.input, output_name)
