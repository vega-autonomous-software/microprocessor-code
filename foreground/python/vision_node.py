import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

from cam_functions.camera import get_camera_frame
from cam_functions.shared_mem import save_to_shared_memory as save_camera
from lidar_functions.lidar import detect_cones_lidar
from lidar_functions.shared_mem import save_to_shared_memory as save_lidar


def connect(name):
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            print(f"[vision_node] {name} connected", flush=True)
            return client
        except Exception as error:
            print(f"[vision_node] {name} waiting: {error}", flush=True)
            time.sleep(2)


def camera_worker(stop):
    client, sequence = connect("camera"), 0
    while not stop.is_set():
        started = time.perf_counter()
        try:
            frame, timestamp = get_camera_frame(client)
            if frame is not None:
                sequence += 1
                save_camera(frame, timestamp, sequence)
        except Exception as error:
            print(f"[vision_node] camera error: {error}", flush=True)
            client = connect("camera")
        stop.wait(max(0.001, 0.033 - (time.perf_counter() - started)))


def lidar_worker(stop):
    client, sequence = connect("LiDAR"), 0
    last_timestamp = None
    while not stop.is_set():
        started = time.perf_counter()
        try:
            _, cones, timestamp = detect_cones_lidar(client)
            unique = timestamp > 0 and timestamp != last_timestamp
            if unique:
                last_timestamp = timestamp
                sequence += 1
                save_lidar(cones, timestamp, sequence)
        except Exception as error:
            print(f"[vision_node] LiDAR error: {error}", flush=True)
            client = connect("LiDAR")
        stop.wait(max(0.001, 0.033 - (time.perf_counter() - started)))


if __name__ == "__main__":
    print("[vision_node] Independent timestamped camera and LiDAR streams", flush=True)
    stop_event = threading.Event()
    workers = [
        threading.Thread(target=camera_worker, args=(stop_event,), daemon=True),
        threading.Thread(target=lidar_worker, args=(stop_event,), daemon=True),
    ]
    for worker in workers:
        worker.start()
    try:
        while all(worker.is_alive() for worker in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("[vision_node] Stopped", flush=True)
