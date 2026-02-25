import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import chirp
import subprocess
import os

def create_sweep_video(output_mp4="sweep_test.mp4", duration=10.0, fs=44100):
    print("[*] 1. 20Hz ~ 20kHz Logarithmic Sweep 오디오 생성 중...")
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # 인간의 청각에 맞춘 로그 스케일(Logarithmic) 주파수 상승
    audio_data = chirp(t, f0=20, f1=20000, t1=duration, method='logarithmic')
    
    # 16bit PCM 포맷으로 변환
    audio_16bit = np.int16(audio_data * 32767)
    wavfile.write("temp_sweep.wav", fs, audio_16bit)
    
    print("[*] 2. FFmpeg를 이용해 블랙 비디오와 합성 중...")
    # 256x192 블랙 화면, 15fps, 10초짜리 비디오 생성
    subprocess.run([
        "ffmpeg", "-y", 
        "-f", "lavfi", "-i", "color=c=black:s=256x192:r=15:d=10",
        "-i", "temp_sweep.wav",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", output_mp4
    ], capture_output=True)
    
    os.remove("temp_sweep.wav")
    print(f"[!] 완료: {output_mp4} 가 생성되었습니다. 오리지널 인코더에 먹여보세요!")

if __name__ == "__main__":
    create_sweep_video()
