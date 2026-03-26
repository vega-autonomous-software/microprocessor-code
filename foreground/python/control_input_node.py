import sys
import os
import time
import socket
import json
import threading
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

from net_utils import TcpBroadcastServer


VEHICLE_NAME = "FSCar"
HOST = "0.0.0.0"
PORT_CONTROL_IN = 82
PORT_ACTUATOR_OUT = 83


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def connect_fsds_forever():
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            client.enableApiControl(True, VEHICLE_NAME)
            print("[control_input_node] Connected to FSDS")
            return client
        except Exception as e:
            print(f"[control_input_node] FSDS not ready: {e}")
            time.sleep(2)


client = connect_fsds_forever()
car_controls = fsds.CarControls()
car_controls.throttle = 0.0
car_controls.brake = 1.0
car_controls.steering = 0.0

tcp_pub = TcpBroadcastServer(bind_host="0.0.0.0", bind_port=PORT_ACTUATOR_OUT)
tcp_pub.start()

latest = {
    "throttle": 0.0,
    "brake": 1.0,
    "steering": 0.0,
    "status": "Listening",
    "timestamp_ms": int(time.time() * 1000),
}


def apply_controls(throttle, brake, steering):
    global client

    car_controls.throttle = throttle
    car_controls.brake = brake
    car_controls.steering = steering

    try:
        client.setCarControls(car_controls, VEHICLE_NAME)
    except Exception as e:
        print(f"[control_input_node] setCarControls failed: {e}")
        latest["status"] = "Reconnecting FSDS..."
        client = connect_fsds_forever()
        client.setCarControls(car_controls, VEHICLE_NAME)


def publish_latest():
    payload = {
        "timestamp_ms": latest["timestamp_ms"],
        "throttle": latest["throttle"],
        "brake": latest["brake"],
        "steering": latest["steering"],
    }
    tcp_pub.send_json(payload)


def handle_client(conn, addr):
    print(f"[control_input_node] Client connected: {addr}")
    latest["status"] = f"Connected: {addr}"

    try:
        file = conn.makefile("r")
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except Exception as e:
                print(f"[control_input_node] Bad JSON: {e} | line={line!r}")
                continue

            throttle = clamp(float(msg.get("throttle", 0.0)), 0.0, 1.0)
            brake = clamp(float(msg.get("brake", 0.0)), 0.0, 1.0)
            steering = clamp(float(msg.get("steering", 0.0)), -1.0, 1.0)

            apply_controls(throttle, brake, steering)

            latest["throttle"] = throttle
            latest["brake"] = brake
            latest["steering"] = steering
            latest["timestamp_ms"] = int(time.time() * 1000)
            latest["status"] = "Command applied"

            publish_latest()

            print(
                f"[control_input_node] "
                f"throttle={throttle:.3f} brake={brake:.3f} steering={steering:.3f}"
            )

    except Exception as e:
        print(f"[control_input_node] Client handler error: {e}")
        latest["status"] = f"Client error: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[control_input_node] Client disconnected: {addr}")


def server_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT_CONTROL_IN))
    server.listen(5)

    print(f"[control_input_node] Listening on {HOST}:{PORT_CONTROL_IN}")

    while True:
        try:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
        except Exception as e:
            print(f"[control_input_node] Accept error: {e}")
            time.sleep(1)


root = tk.Tk()
root.title("Control Input Node")
root.geometry("560x260")
root.configure(bg="#1a1a1a")

title_label = tk.Label(
    root,
    text="Control Input Node",
    font=("Arial", 18, "bold"),
    fg="cyan",
    bg="#1a1a1a"
)
title_label.pack(pady=10)

throttle_var = tk.StringVar(value="Throttle: 0.000")
brake_var = tk.StringVar(value="Brake: 1.000")
steering_var = tk.StringVar(value="Steering: 0.000")
status_var = tk.StringVar(value="Status: Listening")

for var in [throttle_var, brake_var, steering_var, status_var]:
    tk.Label(
        root,
        textvariable=var,
        font=("Arial", 14),
        fg="white",
        bg="#1a1a1a",
        anchor="w"
    ).pack(fill="x", padx=20, pady=5)


def refresh_ui():
    throttle_var.set(f"Throttle: {latest['throttle']:.3f}")
    brake_var.set(f"Brake: {latest['brake']:.3f}")
    steering_var.set(f"Steering: {latest['steering']:.3f}")
    status_var.set(f"Status: {latest['status']}")
    root.after(50, refresh_ui)


def on_close():
    try:
        car_controls.throttle = 0.0
        car_controls.brake = 1.0
        car_controls.steering = 0.0
        client.setCarControls(car_controls, VEHICLE_NAME)
        client.enableApiControl(False, VEHICLE_NAME)
    except Exception:
        pass

    tcp_pub.stop()
    root.destroy()


threading.Thread(target=server_thread, daemon=True).start()
root.protocol("WM_DELETE_WINDOW", on_close)
root.after(100, refresh_ui)
root.mainloop()