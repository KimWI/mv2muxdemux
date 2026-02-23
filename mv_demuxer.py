import os
import sys
import argparse
import subprocess
from PIL import Image

def rgb333_to_888(r, g, b):
    return (r * 255 // 7, g * 255 // 7, b * 255 // 7)

class MV2UltimateDemuxer:
    def __init__(self, input_path, fps):
        self.input_path = input_path
        self.fps = fps
        self.block_size = 16384
        self.temp_dir = "mv2_frames"
        self.palette = [(0, 0, 0)] * 16
        self.base_name = os.path.splitext(os.path.basename(input_path))[0]
        if "mv2" in input_path.lower():
            self.ver = 2
        else:
            self.ver = 1
            self.fps = 12
    
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    # ====================================================================
    # [비디오] 완벽하게 작동했던 VDP 및 팔레트 로직 보존
    # ====================================================================
    def update_palette_30b(self, data):
        for i in range(15):
            b1, b2 = data[i*2], data[i*2+1]
            r, b, g = (b1 >> 4) & 0x07, b1 & 0x07, b2 & 0x07
            self.palette[i+1] = rgb333_to_888(r, g, b)

    def decode_frame(self, vram, idx):
        pgt, ct = vram[:6144], vram[6144:12288]
        img = Image.new('RGB', (256, 192))
        pix = img.load()
        for y in range(192):
            for x in range(256):
                b_idx = ((y // 8) * 32) + (x // 8)
                off = (b_idx * 8) + (y % 8)
                p_byte, c_byte = pgt[off], ct[off]
                fg, bg = c_byte >> 4, c_byte & 0x0F
                color = self.palette[fg] if (p_byte & (1 << (7 - (x % 8)))) else self.palette[bg]
                pix[x, y] = color
        img.save(os.path.join(self.temp_dir, f"f_{idx:05d}.png"))
    # ====================================================================

    def extract_audio_chunk(self, block, f_mp3):
        if self.ver == 2:
            """길이 지시자(Length Indicator)를 통해 쪼개진 MP3 덩어리를 1바이트 유실 없이 추출"""
            if len(block) > 12800:
                size_indicator = block[12800]
                # 지시자 값이 유효한 범위(1~128) 내에 있을 때만 처리
                if 0 < size_indicator <= 128:
                    chunk_size = size_indicator * 32
                    # 12801 오프셋부터 인코더가 지시한 크기만큼만 정확하게 잘라냄
                    if 12801 + chunk_size <= len(block):
                        f_mp3.write(block[12801 : 12801 + chunk_size])
        else:
            f_mp3.write(block[12288:12288+944])

    def run(self):
        raw_mp3 = ""
        header_size = 512

        if self.ver == 2:
            raw_mp3 = f"{self.base_name}_ultimate.mp3"
            header_size = 512 # 크기 512바이트 헤더
        else:
            raw_mp3 = f"{self.base_name}_ultimate.raw"
            header_size = 16384 # 헤더 크기 16384 KB 

        print(f"[*] Demuxing {self.input_path} (Ultimate Math-Verified Mode)...")

        
        with open(self.input_path, 'rb') as f_in, open(raw_mp3, 'wb') as f_mp3:
            f_in.seek(header_size) # 처음 헤더  스킵
            # [블록 0 처리] 512바이트를 빼고 읽어 오프셋 정렬 유지
            if self.ver == 2:
                b0 = f_in.read(self.block_size - header_size)
                if len(b0) >= 12318:
                    self.update_palette_30b(b0[12288:12318])
                self.decode_frame(b0[:12288], 0)
                self.extract_audio_chunk(b0, f_mp3)
                
            
            count = 1
            while True:
                block = f_in.read(self.block_size)
                if len(block) < self.block_size: break
                if self.ver == 2:
                    self.update_palette_30b(block[12288:12318])
                else: 
                    self.update_palette_30b(block[12288+944:12288+944+30])
                self.decode_frame(block[:12288], count)
                # FF FB 헤더를 찾지 않고, 무조건 조각(Chunk) 단위로 무손실 병합
                self.extract_audio_chunk(block, f_mp3)
                
                if count % 100 == 0: print(f"  > Processed Frame {count}")
                count += 1
            if self.ver == 1:
                f_mp3.write(b'\x00' * 16)

        self.mux(raw_mp3)

    def mux(self, mp3):
        if self.ver == 2:
            out = f"{self.base_name}_ultimate.m4a"
        else: 
            out = f"{self.base_name}_ultimate.m4a"
        print("[*] Muxing Audio and Video (Direct Stream Copy)...")
        if self.ver == 2:
            cmd = [
                "ffmpeg", "-y", "-framerate", str(self.fps),
                "-i", os.path.join(self.temp_dir, "f_%05d.png"),
                "-i", mp3,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "scale=-1:576",
                #"-c:a", "copy", "-shortest", out
                "-c:a", "alac", "-shortest", out
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-framerate", str(self.fps),
                "-i", os.path.join(self.temp_dir, "f_%05d.png"),
                "-f", "u8", "-ar", "11328", "-ac", "1", "-i", mp3,
                "-vf", "scale=-1:576",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                #"-c:a", "alac", "-b:a", "128k", "-shortest", out
                "-c:a", "alac", "-shortest", out
            ]
        subprocess.run(cmd)
        print(f"[!] The Reverse Engineering is Complete: {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()
    MV2UltimateDemuxer(args.file, args.fps).run()
