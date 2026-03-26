import sys
import os
import time
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds

from net_utils import TcpBroadcastServer


VEHICLE_NAME = "FSCar"
IMU_NAME = "Imu"
PORT = 81


def connect_fsds_forever():
    while True:
        try:
            client = fsds.FSDSClient()
            client.confirmConnection()
            print("[imu_speed_node] Connected to FSDS")
            return client
        except Exception as e:
            print(f"[imu_speed_node] FSDS not ready: {e}")
            time.sleep(2)


client = connect_fsds_forever()

tcp = TcpBroadcastServer(bind_host="0.0.0.0", bind_port=PORT)
tcp.start()


def get_packet():
    global client
    try:
        state = client.getCarState(VEHICLE_NAME)
        imu = client.getImuData(imu_name=IMU_NAME, vehicle_name=VEHICLE_NAME)
    except Exception:
        client = connect_fsds_forever()
        state = client.getCarState(VEHICLE_NAME)
        imu = client.getImuData(imu_name=IMU_NAME, vehicle_name=VEHICLE_NAME)

    return {
        "timestamp_ms": int(time.time() * 1000),
        "ground_speed_mps": float(state.speed),
        "imu": {
            "angular_velocity": {
                "x": float(imu.angular_velocity.x_val),
                "y": float(imu.angular_velocity.y_val),
                "z": float(imu.angular_velocity.z_val),
            },
            "linear_acceleration": {
                "x": float(imu.linear_acceleration.x_val),
                "y": float(imu.linear_acceleration.y_val),
                "z": float(imu.linear_acceleration.z_val),
            },
            "orientation": {
                "x": float(imu.orientation.x_val),
                "y": float(imu.orientation.y_val),
                "z": float(imu.orientation.z_val),
                "w": float(imu.orientation.w_val),
            },
        },
    }


root = tk.Tk()
root.title("IMU + Ground Speed Node")
root.geometry("760x360")
root.configure(bg="#1a1a1a")

title_label = tk.Label(
    root,
    text="IMU + Ground Speed",
    font=("Arial", 18, "bold"),
    fg="cyan",
    bg="#1a1a1a"
)
title_label.pack(pady=10)

speed_var = tk.StringVar(value="Ground speed: 0.000 m/s")
ang_var = tk.StringVar(value="Angular vel: x=0 y=0 z=0")
lin_var = tk.StringVar(value="Linear acc: x=0 y=0 z=0")
ori_var = tk.StringVar(value="Orientation: x=0 y=0 z=0 w=1")
status_var = tk.StringVar(value="Status: Waiting for FSDS")

for var in [speed_var, ang_var, lin_var, ori_var, status_var]:
    tk.Label(
        root,
        textvariable=var,
        font=("Arial", 13),
        fg="white",
        bg="#1a1a1a",
        anchor="w",
        justify="left"
    ).pack(fill="x", padx=20, pady=6)


def update_ui():
    try:
        data = get_packet()
        speed_var.set(f"Ground speed: {data['ground_speed_mps']:.3f} m/s")

        av = data["imu"]["angular_velocity"]
        la = data["imu"]["linear_acceleration"]
        ori = data["imu"]["orientation"]

        ang_var.set(f"Angular vel: x={av['x']:+.4f}  y={av['y']:+.4f}  z={av['z']:+.4f}")
        lin_var.set(f"Linear acc : x={la['x']:+.4f}  y={la['y']:+.4f}  z={la['z']:+.4f}")
        ori_var.set(f"Orientation: x={ori['x']:+.4f}  y={ori['y']:+.4f}  z={ori['z']:+.4f}  w={ori['w']:+.4f}")

        status_var.set("Status: OK")
        tcp.send_json(data)

    except Exception as e:
        status_var.set(f"Status: Error - {e}")

    root.after(50, update_ui)


def on_close():
    tcp.stop()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.after(100, update_ui)
root.mainloop()