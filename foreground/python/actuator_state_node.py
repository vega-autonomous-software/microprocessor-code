import sys
import os
import time
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

from net_utils import TcpBroadcastServer


VEHICLE_NAME = "FSCar"
PORT = 83

client = fsds.FSDSClient()
client.confirmConnection()

tcp = TcpBroadcastServer(bind_host="0.0.0.0", bind_port=PORT)
tcp.start()


def read_controls():
    controls = client.getCarControls(VEHICLE_NAME)
    return {
        "timestamp_ms": int(time.time() * 1000),
        "throttle": float(controls.throttle),
        "brake": float(controls.brake),
        "steering": float(controls.steering),
    }


root = tk.Tk()
root.title("Actuator State Node")
root.geometry("500x220")
root.configure(bg="#1a1a1a")

title_label = tk.Label(root, text="Actuator State", font=("Arial", 18, "bold"),
                       fg="cyan", bg="#1a1a1a")
title_label.pack(pady=10)

throttle_var = tk.StringVar(value="Throttle: 0.000")
brake_var = tk.StringVar(value="Brake: 0.000")
steering_var = tk.StringVar(value="Steering: 0.000")
status_var = tk.StringVar(value="Status: Running")

for var in [throttle_var, brake_var, steering_var, status_var]:
    tk.Label(root, textvariable=var, font=("Arial", 14),
             fg="white", bg="#1a1a1a", anchor="w").pack(fill="x", padx=20, pady=5)


def update_ui():
    try:
        data = read_controls()
        throttle_var.set(f"Throttle: {data['throttle']:.3f}")
        brake_var.set(f"Brake: {data['brake']:.3f}")
        steering_var.set(f"Steering: {data['steering']:.3f}")
        status_var.set("Status: OK")
        tcp.send_json(data)
    except Exception as e:
        throttle_var.set("Throttle: 0.000")
        brake_var.set("Brake: 0.000")
        steering_var.set("Steering: 0.000")
        status_var.set(f"Status: Error - {e}")

    root.after(50, update_ui)


def on_close():
    tcp.stop()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.after(100, update_ui)
root.mainloop()