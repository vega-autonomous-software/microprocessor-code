import socket
import struct
import subprocess
import platform
import threading
import time

DOCKER_HOST = "localhost"


def send_abs():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((DOCKER_HOST, 83))

    while True:
        timestamp = int(time.time() * 1000)
        abs_value = 0.75

        packet = struct.pack("<Qf", timestamp, abs_value)
        sock.sendall(packet)

        time.sleep(0.1)



def start_video():
    system = platform.system()

    if system == "Darwin":
        cmd = [
            "ffmpeg",
            "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", "1280x720",
            "-i", "0",
            "-c:v", "h264_videotoolbox",
            "-b:v", "5M",
            "-f", "mpegts",
            f"tcp://{DOCKER_HOST}:84"
        ]
    elif system == "Windows":
        cmd = [
            "ffmpeg",
            "-f", "dshow",
            "-framerate", "30",
            "-video_size", "1280x720",
            "-i", 'video="Integrated Camera"',
            "-c:v", "h264_nvenc",
            "-b:v", "5M",
            "-f", "mpegts",
            f"tcp://{DOCKER_HOST}:84"
        ]
    else:
        raise Exception("Unsupported OS")

    subprocess.Popen(cmd)


if __name__ == "__main__":
    threading.Thread(target=send_abs).start()
    start_video()
