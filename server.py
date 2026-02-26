import os
import urllib.request
import http.server
import socketserver
import ssl

# Windows 환경 SSL 인증서 오류 방지
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# FFmpeg.wasm v0.12.x 필수 코어 파일 4가지
FILES = {
    "ffmpeg.js": "https://unpkg.com/@ffmpeg/ffmpeg@0.12.7/dist/umd/ffmpeg.js",
    "814.ffmpeg.js": "https://unpkg.com/@ffmpeg/ffmpeg@0.12.7/dist/umd/814.ffmpeg.js",
    "ffmpeg-core.js": "https://unpkg.com/@ffmpeg/core@0.12.4/dist/umd/ffmpeg-core.js",
    "ffmpeg-core.wasm": "https://unpkg.com/@ffmpeg/core@0.12.4/dist/umd/ffmpeg-core.wasm"
}

print("[*] FFmpeg.wasm 무설치 로컬 구동 준비 중...")
for filename, url in FILES.items():
    if not os.path.exists(filename):
        print(f"  -> 다운로드 중: {filename}")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"  -> 파일 확인 됨: {filename}")

class SecureHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # 브라우저 메모리 공유(SharedArrayBuffer)를 위한 필수 보안 헤더
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

PORT = 8000
with socketserver.TCPServer(("0.0.0.0", PORT), SecureHTTPRequestHandler) as httpd:
    print(f"\n[*] 완벽 준비! 브라우저에서 http://IP:{PORT} 에 접속하세요.")
    httpd.serve_forever()
