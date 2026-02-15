import sys
import time
import os
import subprocess

def convert_video():
    if os.path.exists("video.ts"):
        print("Converting video.ts to video.mp4...", flush=True)
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", "video.ts",
            "-c", "copy",
            "video.mp4"
        ])
        print("Conversion complete.", flush=True)

def main():
    input()  

    print("Logger started", flush=True)

    try:
        while True:
            if os.path.exists("telemetry.bin"):
                size = os.path.getsize("telemetry.bin")
                print(f"Telemetry size: {size} bytes", flush=True)

            if os.path.exists("video.ts"):
                size = os.path.getsize("video.ts")
                print(f"Video size: {size} bytes", flush=True)

            time.sleep(5)

    except KeyboardInterrupt:
        print("Logger shutting down...", flush=True)
        convert_video()

if __name__ == "__main__":
    main()
