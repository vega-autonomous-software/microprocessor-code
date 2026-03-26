import socket
import struct
import json
import threading
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io


HOST = "127.0.0.1"

PORT_IMU = 81
PORT_ACTUATOR = 83
PORT_VISION = 84
PORT_CONTROL = 82


class LengthPrefixedReceiver:
    def __init__(self, host, port, name):
        self.host = host
        self.port = port
        self.name = name
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()

    def connect_forever(self):
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                with self.lock:
                    if self.sock:
                        try:
                            self.sock.close()
                        except Exception:
                            pass
                    self.sock = sock
                    self.connected = True
                print(f"[{self.name}] connected to {self.host}:{self.port}")
                return
            except Exception as e:
                print(f"[{self.name}] connect failed: {e}")
                time.sleep(1)

    def recv_exact(self, n):
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("socket closed")
            data += chunk
        return data

    def recv_packet(self):
        header = self.recv_exact(4)
        length = struct.unpack("<I", header)[0]
        payload = self.recv_exact(length)
        return payload

    def close(self):
        with self.lock:
            self.connected = False
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None


class ControlSender:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()

    def connect_forever(self):
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                with self.lock:
                    if self.sock:
                        try:
                            self.sock.close()
                        except Exception:
                            pass
                    self.sock = sock
                    self.connected = True
                print(f"[control_sender] connected to {self.host}:{self.port}")
                return
            except Exception as e:
                print(f"[control_sender] connect failed: {e}")
                time.sleep(1)

    def send_json_line(self, obj):
        payload = (json.dumps(obj) + "\n").encode("utf-8")
        try:
            with self.lock:
                if not self.sock:
                    raise ConnectionError("not connected")
                self.sock.sendall(payload)
        except Exception:
            self.connected = False
            try:
                if self.sock:
                    self.sock.close()
            except Exception:
                pass
            self.sock = None
            raise

    def close(self):
        with self.lock:
            self.connected = False
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None


class TestConsoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FSDS Test Console")
        self.root.geometry("1500x920")
        self.root.configure(bg="#101010")

        self.latest_imu = None
        self.latest_actuator = None
        self.latest_image = None

        self.imu_status = "Disconnected"
        self.act_status = "Disconnected"
        self.vision_status = "Disconnected"
        self.ctrl_status = "Disconnected"

        self.desired = {
            "throttle": 0.0,
            "brake": 0.0,
            "steering": 0.0,
        }

        self.imu_rx = LengthPrefixedReceiver(HOST, PORT_IMU, "imu_rx")
        self.act_rx = LengthPrefixedReceiver(HOST, PORT_ACTUATOR, "actuator_rx")
        self.vision_rx = LengthPrefixedReceiver(HOST, PORT_VISION, "vision_rx")
        self.ctrl_tx = ControlSender(HOST, PORT_CONTROL)

        self._build_ui()

        threading.Thread(target=self._imu_thread, daemon=True).start()
        threading.Thread(target=self._actuator_thread, daemon=True).start()
        threading.Thread(target=self._vision_thread, daemon=True).start()
        threading.Thread(target=self._control_sender_thread, daemon=True).start()

        self.root.after(50, self.refresh_ui)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="FSDS Multi-Port Test Console",
            font=("Arial", 20, "bold"),
            fg="cyan",
            bg="#101010"
        )
        title.pack(pady=10)

        main = tk.Frame(self.root, bg="#101010")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.Frame(main, bg="#101010")
        left.pack(side="left", fill="both", expand=False)

        right = tk.Frame(main, bg="#101010")
        right.pack(side="right", fill="both", expand=True)

        self._build_controls_panel(left)
        self._build_status_panel(left)
        self._build_imu_panel(left)
        self._build_actuator_panel(left)
        self._build_vision_panel(right)

    def _make_section(self, parent, title_text):
        frame = tk.LabelFrame(
            parent,
            text=title_text,
            fg="cyan",
            bg="#181818",
            font=("Arial", 12, "bold"),
            bd=2
        )
        frame.pack(fill="x", padx=8, pady=8)
        return frame

    def _build_controls_panel(self, parent):
        frame = self._make_section(parent, "Control Output -> Port 82")

        self.throttle_var = tk.DoubleVar(value=0.0)
        self.brake_var = tk.DoubleVar(value=0.0)
        self.steering_var = tk.DoubleVar(value=0.0)

        self.throttle_label = tk.StringVar(value="Throttle: 0.000")
        self.brake_label = tk.StringVar(value="Brake: 0.000")
        self.steering_label = tk.StringVar(value="Steering: 0.000")

        tk.Label(frame, textvariable=self.throttle_label, fg="white", bg="#181818", font=("Arial", 11)).pack(anchor="w", padx=12, pady=(8, 2))
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.throttle_var, orient="horizontal", command=self.on_slider_change).pack(fill="x", padx=12)

        tk.Label(frame, textvariable=self.brake_label, fg="white", bg="#181818", font=("Arial", 11)).pack(anchor="w", padx=12, pady=(10, 2))
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.brake_var, orient="horizontal", command=self.on_slider_change).pack(fill="x", padx=12)

        tk.Label(frame, textvariable=self.steering_label, fg="white", bg="#181818", font=("Arial", 11)).pack(anchor="w", padx=12, pady=(10, 2))
        ttk.Scale(frame, from_=-1.0, to=1.0, variable=self.steering_var, orient="horizontal", command=self.on_slider_change).pack(fill="x", padx=12)

        button_row = tk.Frame(frame, bg="#181818")
        button_row.pack(fill="x", padx=12, pady=12)

        tk.Button(button_row, text="Center Steering", command=self.center_steering, width=15).pack(side="left", padx=4)
        tk.Button(button_row, text="Zero Throttle", command=self.zero_throttle, width=15).pack(side="left", padx=4)
        tk.Button(button_row, text="Full Brake", command=self.full_brake, width=15).pack(side="left", padx=4)

    def _build_status_panel(self, parent):
        frame = self._make_section(parent, "Port Status")

        self.imu_status_var = tk.StringVar(value="81 IMU: Disconnected")
        self.act_status_var = tk.StringVar(value="83 Actuator: Disconnected")
        self.vision_status_var = tk.StringVar(value="84 Vision: Disconnected")
        self.ctrl_status_var = tk.StringVar(value="82 Control TX: Disconnected")

        for var in [self.imu_status_var, self.act_status_var, self.vision_status_var, self.ctrl_status_var]:
            tk.Label(frame, textvariable=var, fg="white", bg="#181818", font=("Arial", 11), anchor="w").pack(fill="x", padx=12, pady=4)

    def _build_imu_panel(self, parent):
        frame = self._make_section(parent, "Port 81 - IMU + Speed")

        self.speed_var = tk.StringVar(value="Ground Speed: ---")
        self.ang_var = tk.StringVar(value="Angular Vel: ---")
        self.lin_var = tk.StringVar(value="Linear Acc: ---")
        self.ori_var = tk.StringVar(value="Orientation: ---")

        for var in [self.speed_var, self.ang_var, self.lin_var, self.ori_var]:
            tk.Label(frame, textvariable=var, fg="white", bg="#181818", font=("Arial", 10), justify="left", anchor="w").pack(fill="x", padx=12, pady=4)

    def _build_actuator_panel(self, parent):
        frame = self._make_section(parent, "Port 83 - Actuator State")

        self.act_throttle_var = tk.StringVar(value="Throttle: ---")
        self.act_brake_var = tk.StringVar(value="Brake: ---")
        self.act_steering_var = tk.StringVar(value="Steering: ---")

        for var in [self.act_throttle_var, self.act_brake_var, self.act_steering_var]:
            tk.Label(frame, textvariable=var, fg="white", bg="#181818", font=("Arial", 11), anchor="w").pack(fill="x", padx=12, pady=4)

    def _build_vision_panel(self, parent):
        frame = tk.LabelFrame(
            parent,
            text="Port 84 - Vision Stream",
            fg="cyan",
            bg="#181818",
            font=("Arial", 12, "bold"),
            bd=2
        )
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.image_label = tk.Label(frame, bg="black")
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

    def on_slider_change(self, _=None):
        self.desired["throttle"] = round(float(self.throttle_var.get()), 3)
        self.desired["brake"] = round(float(self.brake_var.get()), 3)
        self.desired["steering"] = round(float(self.steering_var.get()), 3)

    def center_steering(self):
        self.steering_var.set(0.0)
        self.on_slider_change()

    def zero_throttle(self):
        self.throttle_var.set(0.0)
        self.on_slider_change()

    def full_brake(self):
        self.brake_var.set(1.0)
        self.throttle_var.set(0.0)
        self.on_slider_change()

    def _imu_thread(self):
        while True:
            try:
                self.imu_rx.connect_forever()
                self.imu_status = "Connected"

                while True:
                    payload = self.imu_rx.recv_packet()
                    self.latest_imu = json.loads(payload.decode("utf-8"))
            except Exception as e:
                self.imu_status = f"Disconnected ({e})"
                self.imu_rx.close()
                time.sleep(1)

    def _actuator_thread(self):
        while True:
            try:
                self.act_rx.connect_forever()
                self.act_status = "Connected"

                while True:
                    payload = self.act_rx.recv_packet()
                    self.latest_actuator = json.loads(payload.decode("utf-8"))
            except Exception as e:
                self.act_status = f"Disconnected ({e})"
                self.act_rx.close()
                time.sleep(1)

    def _vision_thread(self):
        while True:
            try:
                self.vision_rx.connect_forever()
                self.vision_status = "Connected"

                while True:
                    payload = self.vision_rx.recv_packet()
                    self.latest_image = payload
            except Exception as e:
                self.vision_status = f"Disconnected ({e})"
                self.vision_rx.close()
                time.sleep(1)

    def _control_sender_thread(self):
        while True:
            try:
                self.ctrl_tx.connect_forever()
                self.ctrl_status = "Connected"

                while True:
                    self.ctrl_tx.send_json_line(self.desired)
                    time.sleep(0.05)
            except Exception as e:
                self.ctrl_status = f"Disconnected ({e})"
                self.ctrl_tx.close()
                time.sleep(1)

    def refresh_ui(self):
        self.throttle_label.set(f"Throttle: {self.desired['throttle']:.3f}")
        self.brake_label.set(f"Brake: {self.desired['brake']:.3f}")
        self.steering_label.set(f"Steering: {self.desired['steering']:.3f}")

        self.imu_status_var.set(f"81 IMU: {self.imu_status}")
        self.act_status_var.set(f"83 Actuator: {self.act_status}")
        self.vision_status_var.set(f"84 Vision: {self.vision_status}")
        self.ctrl_status_var.set(f"82 Control TX: {self.ctrl_status}")

        if self.latest_imu:
            try:
                speed = self.latest_imu.get("ground_speed_mps", 0.0)
                imu = self.latest_imu.get("imu", {})

                av = imu.get("angular_velocity", {})
                la = imu.get("linear_acceleration", {})
                ori = imu.get("orientation", {})

                self.speed_var.set(f"Ground Speed: {speed:.3f} m/s")
                self.ang_var.set(
                    f"Angular Vel: x={av.get('x', 0.0):+.4f}  y={av.get('y', 0.0):+.4f}  z={av.get('z', 0.0):+.4f}"
                )
                self.lin_var.set(
                    f"Linear Acc: x={la.get('x', 0.0):+.4f}  y={la.get('y', 0.0):+.4f}  z={la.get('z', 0.0):+.4f}"
                )
                self.ori_var.set(
                    f"Orientation: x={ori.get('x', 0.0):+.4f}  y={ori.get('y', 0.0):+.4f}  z={ori.get('z', 0.0):+.4f}  w={ori.get('w', 1.0):+.4f}"
                )
            except Exception as e:
                self.speed_var.set(f"IMU parse error: {e}")

        if self.latest_actuator:
            try:
                self.act_throttle_var.set(f"Throttle: {float(self.latest_actuator.get('throttle', 0.0)):.3f}")
                self.act_brake_var.set(f"Brake: {float(self.latest_actuator.get('brake', 0.0)):.3f}")
                self.act_steering_var.set(f"Steering: {float(self.latest_actuator.get('steering', 0.0)):.3f}")
            except Exception as e:
                self.act_throttle_var.set(f"Actuator parse error: {e}")

        if self.latest_image:
            try:
                img = Image.open(io.BytesIO(self.latest_image))
                img.thumbnail((980, 760))
                tk_img = ImageTk.PhotoImage(img)
                self.image_label.configure(image=tk_img)
                self.image_label.image = tk_img
            except Exception as e:
                self.vision_status = f"Image decode error ({e})"

        self.root.after(50, self.refresh_ui)

    def on_close(self):
        try:
            self.ctrl_tx.send_json_line({
                "throttle": 0.0,
                "brake": 1.0,
                "steering": 0.0
            })
        except Exception:
            pass

        self.imu_rx.close()
        self.act_rx.close()
        self.vision_rx.close()
        self.ctrl_tx.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TestConsoleApp(root)
    root.mainloop()