import struct
import json
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import io
import os
import numpy as np
import torch
from ultralytics import YOLO
import cv2
import math
import queue
import mmap
from collections import defaultdict, deque
from ekf_slam import EKFSLAM, MAX_ACTIVE_EKF_LANDMARKS
from autonomous_controller import AutonomousController


SHARED_MEM_DIR_FG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "forground"))
SHARED_MEM_DIR_BG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sharedmemory", "background"))

CAM_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "cam.bin")
IMU_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "ekfin_imu_groundspeed_gyro.bin")
ACT_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "abs_current.bin")
LIDAR_BIN_PATH = os.path.join(SHARED_MEM_DIR_FG, "lid.bin")
CONTROLS_PATH = os.path.join(SHARED_MEM_DIR_BG, "control_instruction.bin")
CAMERA_HORIZONTAL_FOV_DEGREES = 90.0
MIN_GEOMETRY_CONFIRMATIONS = 3
MIN_INITIAL_MAP_LANDMARKS = 4
INITIAL_MAP_STABLE_SCANS = 5
IMU_FRESHNESS_LIMIT_MS = 500
MAX_SENSOR_SYNC_NS = 50_000_000
COLOR_SYNC_WAIT_S = 0.060
PENDING_COLOR_MAX = 8
CAMERA_X_M = -0.30
CAMERA_Y_M = -0.16
LIDAR_X_M = 0.45
LIDAR_Y_M = 0.0
LIDAR_PROJECTION_MARGIN_PX = 10.0
CANDIDATE_MAX_RANGE_M = 12.0
COLOR_MAX_RANGE_M = 10.0
YOLO_BASE_CONFIDENCE = 0.60
YOLO_MIN_CONFIDENCE = 0.45
COLOR_CONFIRM_SCORE = 0.90
COLOR_CONFIRM_MARGIN = 0.35
MIN_COLOR_WINNER_SHARE = 0.75
CANDIDATE_BASE_GATE_M = 0.45
CANDIDATE_MAX_GATE_M = 0.80
CANDIDATE_EXPIRY_SCANS = 20
CANDIDATE_GLOBAL_MAP_CAPACITY = 1200
CANDIDATE_EKF_WINDOW_RADIUS_M = 30.0
CANDIDATE_MAX_POSITION_UNCERTAINTY_M = 0.35
MAX_MAPPING_YAW_RATE_RAD_S = 0.60
MAX_MAPPING_POSITION_SIGMA_M = 0.35
CONE_GRID_CELL_METERS = 1.5
COLOR_INFERENCE_RADIUS_M = 25.0
GLOBAL_PATH_QUERY_RADIUS_M = 25.0
BOUNDARY_NEIGHBOR_RANGE_M = 8.0
BOUNDARY_MAX_ERROR_M = 1.0
ORANGE_GROUP_RANGE_M = 6.0
ORANGE_LOOP_CLOSURE_GATE_M = 3.0
ORANGE_LOOP_CLOSURE_MAX_TRANSLATION_M = 2.5
ORANGE_LOOP_CLOSURE_MAX_YAW_RAD = math.radians(15.0)
ORANGE_LOOP_CLOSURE_RECENT_SCANS = 4
MAP_LOCALIZATION_GATE_M = 1.5
MAP_LOCALIZATION_MAX_TRANSLATION_M = 0.65
MAP_LOCALIZATION_MAX_YAW_RAD = math.radians(5.0)
MAP_LOCALIZATION_MAX_RESIDUAL_M = 0.30
MAP_LOCALIZATION_MIN_MATCHES = 3
MAP_REFERENCE_MIN_AGE_SCANS = 8
MAP_WIDTH_PX = 700
MAP_HEIGHT_PX = 700
MAP_EMBED_SIZE_PX = 440
LIDAR_EMBED_HEIGHT_PX = 320
MAP_MARGIN_PX = 45
MAP_INITIAL_HALF_SPAN_M = 15.0
MAP_EDGE_PADDING_M = 5.0
MAP_EXPANSION_FRACTION = 0.25
UI_BG = "#0d1117"
UI_PANEL = "#161b22"
UI_BORDER = "#30363d"
UI_ACCENT = "#58a6ff"
UI_TEXT = "#f0f6fc"
UI_MUTED = "#8b949e"
UI_BUTTON = "#21262d"
UI_DANGER = "#da3633"
UI_SUCCESS = "#238636"


class TestConsoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FSDS Test Console")
        self.root.geometry("1550x950")
        self.root.configure(bg=UI_BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TScale", background=UI_PANEL, troughcolor=UI_BORDER,
            bordercolor=UI_PANEL, lightcolor=UI_ACCENT, darkcolor=UI_ACCENT,
        )

        self.latest_imu = None
        self.imu_feedback_fresh = False
        self.latest_actuator = None
        self.latest_image = None
        self.shutdown_event = threading.Event()
        self.closing = False

        _model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
        self.yolo_device = 0 if torch.cuda.is_available() else "cpu"
        self.yolo_half = torch.cuda.is_available()
        try:
            self.yolo_model = YOLO(_model_path)
            if self.yolo_half:
                torch.backends.cudnn.benchmark = True
            # Warm-up removes the first-frame latency spike.
            self.yolo_model.predict(
                np.zeros((384, 640, 3), dtype=np.uint8), imgsz=640,
                device=self.yolo_device, half=self.yolo_half, verbose=False,
            )
            print(f"[YOLO] {_model_path} on {self.yolo_device}")
        except Exception as _e:
            self.yolo_model = None
            print(f"[YOLO] failed to load model: {_e}")

        self.imu_status = "Disconnected"
        self.act_status = "Disconnected"
        self.vision_status = "Disconnected"
        self.ctrl_status = "Disconnected"

        self.manual_desired = {
            "throttle": 0.0,
            "brake": 1.0,
            "steering": 0.0,
        }
        self.desired_lock = threading.Lock()
        self.desired = dict(self.manual_desired)

        # Initialize EKF SLAM state estimator
        self.ekf = EKFSLAM()
        self.last_imu_sensor_timestamp_ns = 0
        self.pose_history = deque(maxlen=200)
        self.initial_map_ready = False
        self.map_readiness_reason = "Waiting for confirmed cone geometry"
        self.last_camera_sequence = 0
        self.last_lidar_sequence = 0
        self.last_lidar_timestamp_ns = 0
        self.metrics = defaultdict(float)
        self.cone_candidates = []
        # Global cone map is stored once; this grid makes nearby association
        # constant-time instead of scanning every cone seen during the run.
        self.candidate_by_id = {}
        self.candidate_spatial_index = defaultdict(set)
        self.candidate_scan_index = 0
        self.next_candidate_id = 0
        self.last_geometry_promotion_scan = 0
        # The first reliably observed orange gate becomes a fixed loop-closure
        # reference.  It is deliberately separate from ordinary cone colour
        # inference: orange cones are a distinctive start/finish landmark.
        self.orange_reference_ids = set()
        self.persistent_map_locked = False
        # Grow-to-fit world viewport. Bounds only expand, so global coordinates
        # stay meaningful without the car or mapped cones leaving the display.
        self.map_x_min_m = -MAP_INITIAL_HALF_SPAN_M
        self.map_x_max_m = MAP_INITIAL_HALF_SPAN_M
        self.map_y_min_m = -MAP_INITIAL_HALF_SPAN_M
        self.map_y_max_m = MAP_INITIAL_HALF_SPAN_M

        # Initialize Autonomous Controller with a conservative simulation speed.
        self.controller = AutonomousController(target_speed=1.5)
        self.autonomous_mode = False
        self.autonomy_button_text = tk.StringVar(value="Enable Autonomy")
        self.autonomy_status = "IDLE"

        # Dynamic Tuning variables initialized from self.controller defaults
        self.tune_L_var = tk.DoubleVar(value=self.controller.L)
        self.tune_max_steer_var = tk.DoubleVar(value=math.degrees(self.controller.max_steer_angle))
        self.tune_max_accel_var = tk.DoubleVar(value=self.controller.max_acceleration)
        self.tune_max_decel_var = tk.DoubleVar(value=self.controller.max_deceleration)
        self.tune_k_e_var = tk.DoubleVar(value=self.controller.k_e)
        self.tune_k_s_var = tk.DoubleVar(value=self.controller.k_s)
        self.tune_lookahead_var = tk.DoubleVar(value=self.controller.lookahead_dist)
        self.tune_max_path_length_var = tk.DoubleVar(value=self.controller.max_path_length)
        self.tune_target_speed_var = tk.DoubleVar(value=self.controller.target_speed)

        # LiDAR updates geometry immediately; YOLO adds colour asynchronously.
        self.yolo_input_queue = queue.Queue(maxsize=1)
        self.mapping_queue = queue.Queue(maxsize=128)
        self.yolo_lock = threading.Lock()
        self.sensor_lock = threading.Lock()
        self.lidar_buffer = deque(maxlen=60)
        self.pending_color_results = deque()
        self.latest_cam_cones = []
        self.latest_fused_measurements = []
        self.latest_geometry_measurements = []
        self.latest_cam_img = None
        self.latest_lidar_img = None

        # Keyboard driving state
        self.pressed_keys = {}
        self.using_keyboard = False
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()

        threading.Thread(target=self._shm_reader_thread, daemon=True).start()
        threading.Thread(target=self._shm_writer_thread, daemon=True).start()
        threading.Thread(target=self._yolo_worker_thread, daemon=True).start()

        self.root.after(50, self.refresh_ui)
        self.root.after(100, self._refresh_embedded_map)

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="FSDS SHM Test Console",
            font=("Segoe UI", 20, "bold"),
            fg=UI_TEXT,
            bg=UI_BG,
        )
        title.pack(pady=5)

        main = tk.Frame(self.root, bg=UI_BG)
        main.pack(fill="both", expand=True, padx=10, pady=5)

        left = tk.Frame(main, bg=UI_BG)
        left.pack(side="left", fill="both", expand=False)
        right = tk.Frame(main, bg=UI_BG)
        right.pack(side="right", fill="both", expand=True)

        # Controls/status share a column group so the map can use their free space.
        left_primary = tk.Frame(left, bg=UI_BG)
        left_primary.pack(side="left", fill="both", expand=False)
        info_row = tk.Frame(left_primary, bg=UI_BG)
        info_row.pack(side="top", fill="x", expand=False)

        col1 = tk.Frame(info_row, bg=UI_BG)
        col1.pack(side="left", fill="both", expand=False, padx=5)

        col2 = tk.Frame(info_row, bg=UI_BG)
        col2.pack(side="left", fill="both", expand=False, padx=5)

        col3 = tk.Frame(left, bg=UI_BG)
        col3.pack(side="left", fill="both", expand=False, padx=5)

        self._build_controls_panel(col1)
        self._build_slam_panel(col1)

        self._build_status_panel(col2)
        self._build_imu_panel(col2)
        self._build_actuator_panel(col2)

        self._build_tuning_panel(col3)
        self._build_lidar_panel(left_primary)

        self._build_vision_panel(right)

    def _build_slam_panel(self, parent):
        frame = self._make_section(parent, "EKF SLAM Measurements (Range, Bearing)")
        self.slam_text_var = tk.StringVar(value="No detections yet")
        tk.Label(
            frame,
            textvariable=self.slam_text_var,
            fg=UI_ACCENT,
            bg=UI_PANEL,
            font=("Consolas", 10),
            justify="left",
            anchor="w"
        ).pack(fill="x", padx=12, pady=(8, 4))

    def _make_section(self, parent, title_text):
        frame = tk.LabelFrame(
            parent, text=title_text, fg=UI_ACCENT, bg=UI_PANEL,
            font=("Segoe UI", 11, "bold"), bd=0, relief="flat",
            highlightthickness=1, highlightbackground=UI_BORDER,
        )
        frame.pack(fill="x", padx=8, pady=8)
        return frame

    @staticmethod
    def _make_button(parent, text, command, width=15, danger=False):
        return tk.Button(
            parent, text=text, command=command, width=width,
            bg=UI_DANGER if danger else UI_BUTTON, fg=UI_TEXT,
            activebackground="#f85149" if danger else UI_BORDER,
            activeforeground=UI_TEXT, relief="flat", bd=0,
            font=("Segoe UI", 9), cursor="hand2", padx=6, pady=5,
        )

    def _build_controls_panel(self, parent):
        frame = self._make_section(parent, "Control Output -> SHM")

        self.throttle_var = tk.DoubleVar(value=0.0)
        self.brake_var = tk.DoubleVar(value=1.0)
        self.steering_var = tk.DoubleVar(value=0.0)

        self.throttle_label = tk.StringVar(value="Throttle: 0.000")
        self.brake_label = tk.StringVar(value="Brake: 0.000")
        self.steering_label = tk.StringVar(value="Steering: 0.000")

        tk.Label(frame, textvariable=self.throttle_label, fg=UI_TEXT, bg=UI_PANEL, font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(8, 2))
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.throttle_var, orient="horizontal", command=self.on_slider_change).pack(fill="x", padx=12)
        tk.Label(frame, textvariable=self.brake_label, fg=UI_TEXT, bg=UI_PANEL, font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(10, 2))
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self.brake_var, orient="horizontal", command=self.on_slider_change).pack(fill="x", padx=12)
        tk.Label(frame, textvariable=self.steering_label, fg=UI_TEXT, bg=UI_PANEL, font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(10, 2))
        ttk.Scale(frame, from_=-1.0, to=1.0, variable=self.steering_var, orient="horizontal", command=self.on_slider_change).pack(fill="x", padx=12)

        button_row = tk.Frame(frame, bg=UI_PANEL)
        button_row.pack(fill="x", padx=12, pady=12)
        self._make_button(button_row, "Center Steering", self.center_steering).pack(side="left", padx=4)
        self._make_button(button_row, "Zero Throttle", self.zero_throttle).pack(side="left", padx=4)
        self._make_button(button_row, "Full Brake", self.full_brake, danger=True).pack(side="left", padx=4)

        self.autonomy_button = tk.Button(
            frame,
            textvariable=self.autonomy_button_text,
            command=self.toggle_autonomy,
            width=24,
            bg=UI_SUCCESS,
            fg=UI_TEXT,
            activebackground="#2f7a49",
            activeforeground=UI_TEXT,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 9),
        )
        self.autonomy_button.pack(pady=(0, 10))

        # Keyboard drive guide
        guide_text = (
            "Keyboard Driving Controls (Focus this window):\n"
            "  - Throttle: W / Up Arrow (limit: 0.40)\n"
            "  - Steering: A/D or Left/Right Arrow\n"
            "  - Brake: S / Down Arrow\n"
            "  - E-Brake: Space\n"
            "  - Toggle Auto Mode: P"
        )
        tk.Label(
            frame,
            text=guide_text,
            fg=UI_MUTED,
            bg=UI_PANEL,
            font=("Segoe UI", 9),
            justify="left",
            anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 8))

    def _build_status_panel(self, parent):
        frame = self._make_section(parent, "SHM Status")

        self.imu_status_var = tk.StringVar(value="IMU: Disconnected")
        self.act_status_var = tk.StringVar(value="Actuator: Disconnected")
        self.vision_status_var = tk.StringVar(value="Vision: Disconnected")
        self.ctrl_status_var = tk.StringVar(value="Control TX: Writing")

        for var in [self.imu_status_var, self.act_status_var, self.vision_status_var, self.ctrl_status_var]:
            tk.Label(frame, textvariable=var, fg=UI_TEXT, bg=UI_PANEL, font=("Segoe UI", 10), anchor="w").pack(fill="x", padx=12, pady=4)

    def _build_imu_panel(self, parent):
        frame = self._make_section(parent, "SHM - IMU + Speed")

        self.speed_var = tk.StringVar(value="Ground Speed: ---")
        self.ang_var = tk.StringVar(value="Angular Vel: ---")
        self.lin_var = tk.StringVar(value="Linear Acc: ---")
        self.ori_var = tk.StringVar(value="Orientation: ---")

        for var in [self.speed_var, self.ang_var, self.lin_var, self.ori_var]:
            tk.Label(frame, textvariable=var, fg=UI_TEXT, bg=UI_PANEL, font=("Segoe UI", 9), justify="left", anchor="w").pack(fill="x", padx=12, pady=4)

    def _build_actuator_panel(self, parent):
        frame = self._make_section(parent, "SHM - Actuator State")

        self.act_throttle_var = tk.StringVar(value="Throttle: ---")
        self.act_brake_var = tk.StringVar(value="Brake: ---")
        self.act_steering_var = tk.StringVar(value="Steering: ---")

        for var in [self.act_throttle_var, self.act_brake_var, self.act_steering_var]:
            tk.Label(frame, textvariable=var, fg=UI_TEXT, bg=UI_PANEL, font=("Segoe UI", 10), anchor="w").pack(fill="x", padx=12, pady=4)

    def _build_vision_panel(self, parent):
        camera_frame = tk.LabelFrame(
            parent,
            text="SHM Camera Stream",
            fg=UI_ACCENT, bg=UI_PANEL, font=("Segoe UI", 11, "bold"),
            bd=0, highlightthickness=1, highlightbackground=UI_BORDER,
        )
        camera_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.camera_label = tk.Label(camera_frame, bg="#080b10")
        self.camera_label.pack(fill="both", expand=True, padx=8, pady=8)

        map_frame = tk.LabelFrame(
            parent,
            text="EKF SLAM Adaptive World Map",
            fg=UI_ACCENT, bg=UI_PANEL, font=("Segoe UI", 11, "bold"),
            bd=0, highlightthickness=1, highlightbackground=UI_BORDER,
        )
        map_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.map_label = tk.Label(map_frame, bg="#080b10")
        self.map_label.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_tuning_panel(self, parent):
        frame = self._make_section(parent, "Autonomous Controller Tuning")

        # Helper to add a slider
        def add_tuning_slider(label_var, double_var, from_val, to_val):
            tk.Label(frame, textvariable=label_var, fg=UI_TEXT, bg=UI_PANEL, font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(4, 1))
            scale = ttk.Scale(frame, from_=from_val, to=to_val, variable=double_var, orient="horizontal")
            scale.pack(fill="x", padx=12, pady=(0, 4))
            return scale

        self.lbl_L = tk.StringVar()
        self.lbl_max_steer = tk.StringVar()
        self.lbl_max_accel = tk.StringVar()
        self.lbl_max_decel = tk.StringVar()
        self.lbl_k_e = tk.StringVar()
        self.lbl_k_s = tk.StringVar()
        self.lbl_lookahead = tk.StringVar()
        self.lbl_max_path_length = tk.StringVar()
        self.lbl_target_speed = tk.StringVar()

        # Add sliders
        add_tuning_slider(self.lbl_L, self.tune_L_var, 1.0, 3.0)
        add_tuning_slider(self.lbl_max_steer, self.tune_max_steer_var, 10.0, 90.0)
        add_tuning_slider(self.lbl_max_accel, self.tune_max_accel_var, 1.0, 20.0)
        add_tuning_slider(self.lbl_max_decel, self.tune_max_decel_var, 1.0, 20.0)
        add_tuning_slider(self.lbl_k_e, self.tune_k_e_var, 0.1, 10.0)
        add_tuning_slider(self.lbl_k_s, self.tune_k_s_var, 0.1, 10.0)
        add_tuning_slider(self.lbl_lookahead, self.tune_lookahead_var, 0.5, 10.0)
        add_tuning_slider(self.lbl_max_path_length, self.tune_max_path_length_var, 2.0, 35.0)
        add_tuning_slider(self.lbl_target_speed, self.tune_target_speed_var, 0.5, 10.0)

        # Set trace on all DoubleVars so they update self.controller immediately on any change!
        for var in [self.tune_L_var, self.tune_max_steer_var, self.tune_max_accel_var, self.tune_max_decel_var,
                    self.tune_k_e_var, self.tune_k_s_var, self.tune_lookahead_var,
                    self.tune_max_path_length_var, self.tune_target_speed_var]:
            var.trace_add("write", self.on_tune_change)

        # Initialize labels and controller
        self.on_tune_change()

        # Reset button
        btn_reset = self._make_button(frame, "Reset to Defaults", self.reset_tune_defaults, width=20)
        btn_reset.pack(pady=10)

    def on_tune_change(self, *args):
        if not hasattr(self, 'controller'):
            return
        try:
            L = float(self.tune_L_var.get())
            max_steer = float(self.tune_max_steer_var.get())
            max_accel = float(self.tune_max_accel_var.get())
            max_decel = float(self.tune_max_decel_var.get())
            k_e = float(self.tune_k_e_var.get())
            k_s = float(self.tune_k_s_var.get())
            lookahead = float(self.tune_lookahead_var.get())
            max_path_length = float(self.tune_max_path_length_var.get())
            target_speed = float(self.tune_target_speed_var.get())
        except (tk.TclError, ValueError):
            return

        self.controller.L = L
        self.controller.max_steer_angle = math.radians(max_steer)
        self.controller.max_acceleration = max_accel
        self.controller.max_deceleration = max_decel
        self.controller.k_e = k_e
        self.controller.k_s = k_s
        self.controller.lookahead_dist = lookahead
        self.controller.max_path_length = max(0.1, max_path_length)
        self.controller.target_speed = target_speed

        # Also update label texts dynamically
        self.lbl_L.set(f"Wheelbase (L): {L:.2f} m")
        self.lbl_max_steer.set(f"Max Steer Angle: {max_steer:.1f}°")
        self.lbl_max_accel.set(f"Max Accel: {max_accel:.1f} m/s²")
        self.lbl_max_decel.set(f"Max Decel: {max_decel:.1f} m/s²")
        self.lbl_k_e.set(f"Stanley k_e: {k_e:.2f}")
        self.lbl_k_s.set(f"Stanley k_s: {k_s:.2f}")
        self.lbl_lookahead.set(f"Lookahead Dist: {lookahead:.2f} m")
        self.lbl_max_path_length.set(f"Purple Path Max Length: {max_path_length:.1f} m")
        self.lbl_target_speed.set(f"Target Speed: {target_speed:.2f} m/s")

    def reset_tune_defaults(self):
        defaults = AutonomousController(target_speed=1.5)
        self.tune_L_var.set(defaults.L)
        self.tune_max_steer_var.set(math.degrees(defaults.max_steer_angle))
        self.tune_max_accel_var.set(defaults.max_acceleration)
        self.tune_max_decel_var.set(defaults.max_deceleration)
        self.tune_k_e_var.set(defaults.k_e)
        self.tune_k_s_var.set(defaults.k_s)
        self.tune_lookahead_var.set(defaults.lookahead_dist)
        self.tune_max_path_length_var.set(defaults.max_path_length)
        self.tune_target_speed_var.set(defaults.target_speed)
        self.on_tune_change()

    def _build_lidar_panel(self, parent):
        """Use the lower-left dashboard space for the LiDAR cone projection."""
        frame = tk.LabelFrame(
            parent, text="LiDAR Cone View",
            fg=UI_ACCENT, bg=UI_PANEL, font=("Segoe UI", 11, "bold"),
            bd=0, highlightthickness=1, highlightbackground=UI_BORDER,
        )
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        frame.configure(width=MAP_EMBED_SIZE_PX, height=LIDAR_EMBED_HEIGHT_PX)
        frame.pack_propagate(False)
        self.lidar_label = tk.Label(frame, bg="#080b10")
        self.lidar_label.pack(fill="both", expand=True, padx=8, pady=8)

    def _refresh_embedded_map(self):
        """Render the map independently at 10 Hz to protect video responsiveness."""
        if self.shutdown_event.is_set():
            return
        self.root.after(100, self._refresh_embedded_map)
        try:
            map_image = Image.fromarray(self._draw_ekf_map())
            available = min(self.map_label.winfo_width(), self.map_label.winfo_height())
            display_size = min(MAP_EMBED_SIZE_PX, available if available > 100 else 420)
            map_image = map_image.resize(
                (display_size, display_size), Image.Resampling.LANCZOS,
            )
            tk_map = ImageTk.PhotoImage(map_image)
            self.map_label.configure(image=tk_map)
            self.map_label.image = tk_map
        except Exception as error:
            print(f"[Dashboard] Error updating embedded EKF map: {error}")

    @staticmethod
    def _display_sensor_image(label, array, fallback_size):
        """Fit a sensor image inside its card without changing its aspect ratio."""
        image = Image.fromarray(array)
        width, height = label.winfo_width(), label.winfo_height()
        limit = (
            width if width > 100 else fallback_size[0],
            height if height > 100 else fallback_size[1],
        )
        image.thumbnail(limit, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo

    def _set_desired(self, command):
        """Validate and select the command that will be sent to foreground."""
        throttle = float(command["throttle"])
        brake = float(command["brake"])
        steering = float(command["steering"])

        if not all(math.isfinite(value) for value in (throttle, brake, steering)):
            raise ValueError("Control command contains a non-finite value")

        selected = {
            "throttle": max(0.0, min(1.0, throttle)),
            "brake": max(0.0, min(1.0, brake)),
            "steering": max(-1.0, min(1.0, steering)),
        }
        with self.desired_lock:
            self.desired.update(selected)

    def _get_desired(self):
        with self.desired_lock:
            return dict(self.desired)

    def _disable_autonomy(self, reason):
        """Return to manual mode with a safe full-brake command."""
        self.autonomous_mode = False
        self.using_keyboard = False
        self.manual_desired.update({
            "throttle": 0.0,
            "brake": 1.0,
            "steering": 0.0,
        })
        self._set_desired(self.manual_desired)
        self._update_autonomy_ui()
        print(f"[Dashboard] Autonomous mode disabled: {reason}")

    def _update_autonomy_ui(self):
        if self.autonomous_mode:
            self.autonomy_button_text.set("Disable Autonomy")
            if hasattr(self, "autonomy_button"):
                self.autonomy_button.configure(bg=UI_DANGER, activebackground="#f85149")
        else:
            self.autonomy_button_text.set("Enable Autonomy")
            if hasattr(self, "autonomy_button"):
                self.autonomy_button.configure(bg=UI_SUCCESS, activebackground="#2ea043")

    def _set_autonomy_status(self, status):
        if status != getattr(self, "autonomy_status", None):
            self.autonomy_status = status
            print(f"[Autonomy] {status}", flush=True)

    def toggle_autonomy(self):
        """Single toggle used by both the UI button and the P key."""
        self.autonomous_mode = not self.autonomous_mode
        if self.autonomous_mode:
            self.using_keyboard = False
            if hasattr(self, "controller"):
                self.controller.path_started_at = None
            self.steering_var.set(0.0)
            self.throttle_var.set(0.0)
            self.brake_var.set(1.0)
            self.on_slider_change()
            self._set_autonomy_status("ENABLED: evaluating map")
        else:
            self._set_desired(self.manual_desired)
            self._set_autonomy_status("IDLE")
        self._update_autonomy_ui()
        print(f"[Dashboard] Autonomous Mode: {self.autonomous_mode}")

    def on_slider_change(self, _=None):
        self.manual_desired["throttle"] = round(float(self.throttle_var.get()), 3)
        self.manual_desired["brake"] = round(float(self.brake_var.get()), 3)
        self.manual_desired["steering"] = round(float(self.steering_var.get()), 3)
        if not self.autonomous_mode or self.using_keyboard:
            self._set_desired(self.manual_desired)

    def center_steering(self):
        self.steering_var.set(0.0)
        self.on_slider_change()

    def zero_throttle(self):
        self.throttle_var.set(0.0)
        self.on_slider_change()

    def full_brake(self):
        if self.autonomous_mode:
            self._disable_autonomy("Manual emergency brake")
        self.brake_var.set(1.0)
        self.throttle_var.set(0.0)
        self.on_slider_change()

    def _shm_reader_thread(self):
        while not self.shutdown_event.is_set():
            try:
                if os.path.exists(IMU_BIN_PATH):
                    try:
                        with open(IMU_BIN_PATH, "rb") as f:
                            if os.fstat(f.fileno()).st_size >= 60:
                                ram = mmap.mmap(f.fileno(), 60, access=mmap.ACCESS_READ)
                                values = struct.unpack("<QQ11f", ram[0:60])
                                t, sensor_t, spd, ax, ay, az, lx, ly, lz, ox, oy, oz, ow = values
                                self.latest_imu = {
                                    "timestamp_ms": t,
                                    "sensor_timestamp_ns": sensor_t,
                                    "ground_speed_mps": spd,
                                    "imu": {
                                        "angular_velocity": {"x": ax, "y": ay, "z": az},
                                        "linear_acceleration": {"x": lx, "y": ly, "z": lz},
                                        "orientation": {"x": ox, "y": oy, "z": oz, "w": ow}
                                    }
                                }
                                self.imu_status = "Reading SHM (mmap)"
                                ram.close()
                    except Exception:
                        pass
                    
                if os.path.exists(ACT_BIN_PATH):
                    try:
                        with open(ACT_BIN_PATH, "rb") as f:
                            if os.fstat(f.fileno()).st_size >= 20:
                                ram = mmap.mmap(f.fileno(), 20, access=mmap.ACCESS_READ)
                                t, th, br, st = struct.unpack("<Qfff", ram[0:20])
                                self.latest_actuator = {
                                    "timestamp_ms": t,
                                    "throttle": th,
                                    "brake": br,
                                    "steering": st
                                }
                                self.act_status = "Fresh feedback (mmap)"
                                ram.close()
                    except Exception:
                        pass

                camera = self._read_camera()
                if camera and camera[2] != self.last_camera_sequence:
                    image, timestamp, self.last_camera_sequence = camera
                    with self.yolo_lock:
                        self.latest_cam_img = image.copy()
                    if self.yolo_input_queue.full():
                        self.yolo_input_queue.get_nowait()
                        self.metrics["queue_drops"] += 1
                    self.yolo_input_queue.put_nowait((image, timestamp))
                    self.metrics["camera_frames"] += 1

                lidar = self._read_lidar()
                if lidar and lidar[2] != self.last_lidar_sequence:
                    timestamp, cones, self.last_lidar_sequence = lidar
                    if timestamp <= 0 or timestamp == self.last_lidar_timestamp_ns:
                        self.metrics["lidar_duplicates"] += 1
                    else:
                        self.last_lidar_timestamp_ns = timestamp
                        with self.sensor_lock:
                            self.lidar_buffer.append((timestamp, cones))
                        observations = self._lidar_observations(cones, timestamp)
                        self._queue_mapping("geometry", observations)
                        with self.yolo_lock:
                            self.latest_lidar_img = np.zeros((540, 700, 3), dtype=np.uint8)
                            self.latest_geometry_measurements = observations
                        self.metrics["lidar_frames"] += 1
                        self.metrics["lidar_cones"] += len(cones)
                        self.metrics["geometry_observations"] += len(observations)
                        self.vision_status = "Independent camera/LiDAR SHM"

                # Resolve pending colours when a future scan arrives or expires.
                self._resolve_pending_colors()
            except Exception:
                pass
            self.shutdown_event.wait(0.033)

    @staticmethod
    def _read_camera():
        if not os.path.exists(CAM_BIN_PATH):
            return None
        with open(CAM_BIN_PATH, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < 48:
                return None
            ram = mmap.mmap(stream.fileno(), size, access=mmap.ACCESS_READ)
            header = bytes(ram[:48])
            h, w, channels, _, timestamp, sequence = struct.unpack("QQQQQQ", header)
            image_size = int(h * w * channels)
            if not sequence or size < 48 + image_size:
                ram.close()
                return None
            payload = bytes(ram[48:48 + image_size])
            stable = header == bytes(ram[:48])
            ram.close()
        if not stable:
            return None
        image = np.frombuffer(payload, np.uint8).reshape(int(h), int(w), int(channels))
        return image, int(timestamp), int(sequence)

    @staticmethod
    def _read_lidar():
        if not os.path.exists(LIDAR_BIN_PATH):
            return None
        with open(LIDAR_BIN_PATH, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size < 24:
                return None
            ram = mmap.mmap(stream.fileno(), size, access=mmap.ACCESS_READ)
            header = bytes(ram[:24])
            count, timestamp, sequence = struct.unpack("QQQ", header)
            if not sequence or size < 24 + count * 16:
                ram.close()
                return None
            payload = bytes(ram[24:24 + count * 16])
            stable = header == bytes(ram[:24])
            ram.close()
        if not stable:
            return None
        points = np.frombuffer(payload, dtype=np.float64).reshape(-1, 2)
        scale, center_x, center_y = (540 - 80) / 20.0, 350, 500
        cones = [(int(center_x + y * scale), int(center_y - x * scale), math.hypot(x, y), y, x)
                 for x, y in points]
        return int(timestamp), cones, int(sequence)

    def _queue_mapping(self, kind, measurements):
        if not measurements:
            return
        try:
            self.mapping_queue.put_nowait((kind, measurements))
        except queue.Full:
            self.metrics["mapping_drops"] += 1

    @staticmethod
    def _lidar_observations(cones, timestamp):
        observations = []
        for px, py, _, left, forward in cones:
            vehicle_x, vehicle_y = forward + LIDAR_X_M, left + LIDAR_Y_M
            distance = math.hypot(vehicle_x, vehicle_y)
            if 0.5 <= distance <= CANDIDATE_MAX_RANGE_M:
                observations.append({
                    "label": "unknown", "color": "unknown", "conf": 0.0,
                    "cam_box": None, "box_height": 0.0,
                    "lidar_pixel": [int(px), int(py)], "range": distance,
                    "bearing": math.atan2(vehicle_y, vehicle_x),
                    "lidar_timestamp_ns": int(timestamp),
                })
        return observations

    def _shm_writer_thread(self):
        os.makedirs(SHARED_MEM_DIR_BG, exist_ok=True)
        BINARY_FORMAT = "<Qfff"
        STRUCT_SIZE = struct.calcsize(BINARY_FORMAT)

        if not os.path.exists(CONTROLS_PATH) or os.path.getsize(CONTROLS_PATH) != STRUCT_SIZE:
            with open(CONTROLS_PATH, "wb") as f:
                f.write(b"\x00" * STRUCT_SIZE)

        with open(CONTROLS_PATH, "r+b") as f:
            ram = mmap.mmap(f.fileno(), STRUCT_SIZE)
            try:
                while not self.shutdown_event.is_set():
                    try:
                        if self.autonomous_mode and not self.using_keyboard:
                            # The candidate map is the persistent global map.
                            # Query only its nearby grid cells for planning, so
                            # lap length cannot make this loop slower.
                            car_x, car_y = float(self.ekf.x[0]), float(self.ekf.x[1])
                            ekf_state = []
                            for candidate in self._nearby_candidates(
                                car_x, car_y, GLOBAL_PATH_QUERY_RADIUS_M,
                            ):
                                color = str(candidate.get("color") or "unknown").lower()
                                if (
                                    candidate.get("geometry_confirmed")
                                    and any(name in color for name in ("blue", "yellow", "orange"))
                                ):
                                    ekf_state.append({
                                        "x": float(candidate["x"]),
                                        "y": float(candidate["y"]),
                                        "color": color,
                                    })

                            speed, _ = self._current_motion()
                            thr, strng, brk = self.controller.compute_commands(
                                ekf_state, self.ekf.x[0], self.ekf.x[1], self.ekf.x[2], speed,
                            )
                            self._set_desired({
                                "throttle": thr,
                                "steering": strng,
                                "brake": brk,
                            })
                            self._set_autonomy_status(self.controller.last_status)
                    except Exception as error:
                        self._set_desired({"throttle": 0.0, "brake": 1.0, "steering": 0.0})
                        self._set_autonomy_status(f"ERROR: {type(error).__name__}: {error}")
                        traceback.print_exc()

                    command = self._get_desired()
                    packed = struct.pack(
                        BINARY_FORMAT,
                        int(time.time() * 1000),
                        command["throttle"],
                        command["brake"],
                        command["steering"],
                    )
                    ram[0:STRUCT_SIZE] = packed
                    self.ctrl_status = (
                        f"Writing (mmap) | {self.autonomy_status}"
                        if self.autonomous_mode else "Writing (mmap)"
                    )
                    self.shutdown_event.wait(0.05)
            finally:
                # Always leave a fresh full-brake command for foreground.
                ram[0:STRUCT_SIZE] = struct.pack(
                    BINARY_FORMAT, int(time.time() * 1000), 0.0, 1.0, 0.0
                )
                ram.flush()
                ram.close()

    def _yolo_worker_thread(self):
        while not self.shutdown_event.is_set():
            try:
                cam_img, camera_timestamp_ns = self.yolo_input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                started = time.perf_counter()
                results = self.yolo_model.predict(
                    cam_img, imgsz=640, device=self.yolo_device,
                    half=self.yolo_half, verbose=False,
                )[0] if self.yolo_model else None
                inference_s = time.perf_counter() - started
                self.metrics["yolo_frames"] += 1
                self.metrics["inference_s"] += inference_s

                speed, yaw_rate = self._current_motion()
                dynamic_conf = max(
                    YOLO_MIN_CONFIDENCE,
                    YOLO_BASE_CONFIDENCE - min(0.15, 0.02 * speed + 0.5 * inference_s),
                )
                cam_cones = self._camera_cones(results, cam_img.shape[1], dynamic_conf)
                self._queue_pending_color(
                    cam_cones, camera_timestamp_ns, cam_img.shape[1],
                    speed, yaw_rate,
                )

                with self.yolo_lock:
                    self.latest_cam_cones = cam_cones
            except Exception as e:
                print(f"[YOLO Thread] Error: {e}")
            finally:
                self.yolo_input_queue.task_done()

    def _current_motion(self):
        imu = self.latest_imu or {}
        fresh = int(time.time() * 1000) - int(imu.get("timestamp_ms", 0)) <= IMU_FRESHNESS_LIMIT_MS
        if not fresh:
            return 0.0, 0.0
        return (
            abs(float(imu.get("ground_speed_mps", 0.0))),
            float(imu.get("imu", {}).get("angular_velocity", {}).get("z", 0.0)),
        )

    def _camera_cones(self, results, image_width, min_confidence):
        if results is None:
            return []
        raw = []
        self.metrics["raw_boxes"] += len(results.boxes)
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            if confidence < min_confidence or x1 <= 5 or x2 >= image_width - 5:
                self.metrics["box_rejects"] += 1
                continue
            label = results.names.get(int(box.cls[0]), str(int(box.cls[0])))
            raw.append(((x1 + x2) / 2.0, x1, y1, x2, y2, label, confidence))

        kept = []
        for cone in sorted(raw, key=lambda item: item[6], reverse=True):
            if all(self._box_iou(cone, other) <= 0.3 for other in kept):
                kept.append(cone)
            else:
                self.metrics["box_rejects"] += 1
        return kept

    @staticmethod
    def _box_iou(first, second):
        _, ax1, ay1, ax2, ay2, _, _ = first
        _, bx1, by1, bx2, by2, _, _ = second
        overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
        return overlap / union if union > 0.0 else 0.0

    def _queue_pending_color(self, camera_cones, timestamp, image_width, speed, yaw_rate):
        """Hold YOLO output briefly so a closer future LiDAR scan can arrive."""
        self._resolve_pending_colors()
        entry = {
            "camera_cones": camera_cones,
            "camera_timestamp": int(timestamp),
            "image_width": int(image_width),
            "speed": float(speed),
            "yaw_rate": float(yaw_rate),
            "deadline": time.monotonic() + COLOR_SYNC_WAIT_S,
        }
        dropped = False
        with self.sensor_lock:
            if len(self.pending_color_results) >= PENDING_COLOR_MAX:
                self.pending_color_results.popleft()
                dropped = True
            self.pending_color_results.append(entry)
        self.metrics["sync_rejects"] += int(dropped)
        self._resolve_pending_colors()

    @staticmethod
    def _pending_lidar_match(entry, scans, now):
        """Return (finished, closest scan); wait unless future data or expiry exists."""
        camera_timestamp = entry["camera_timestamp"]
        future_seen = any(timestamp >= camera_timestamp for timestamp, _ in scans)
        if not future_seen and now < entry["deadline"]:
            return False, None
        if not scans:
            return True, None
        nearest = min(scans, key=lambda sample: abs(sample[0] - camera_timestamp))
        if abs(nearest[0] - camera_timestamp) > MAX_SENSOR_SYNC_NS:
            nearest = None
        return True, nearest

    def _resolve_pending_colors(self):
        """Resolve ready colour results without blocking camera or LiDAR workers."""
        now, ready = time.monotonic(), []
        with self.sensor_lock:
            scans = tuple(self.lidar_buffer)
            waiting = deque()
            while self.pending_color_results:
                entry = self.pending_color_results.popleft()
                finished, nearest = self._pending_lidar_match(entry, scans, now)
                if finished:
                    ready.append((entry, nearest))
                else:
                    waiting.append(entry)
            self.pending_color_results = waiting

        for entry, nearest in ready:
            if nearest is None:
                self.metrics["sync_rejects"] += 1
                continue
            lidar_timestamp, lidar_cones = nearest
            observations = self._fuse_camera_color(
                entry["camera_cones"], lidar_cones,
                entry["camera_timestamp"], lidar_timestamp,
                entry["image_width"], entry["speed"], entry["yaw_rate"],
            )
            self._queue_mapping("color", observations)
            with self.yolo_lock:
                self.latest_fused_measurements = observations

    def _nearest_lidar(self, camera_timestamp):
        """Return the actual closest LiDAR scan, never merely the latest one."""
        with self.sensor_lock:
            if not self.lidar_buffer:
                return None
            timestamp, cones = min(
                self.lidar_buffer, key=lambda sample: abs(sample[0] - camera_timestamp)
            )
        if abs(timestamp - camera_timestamp) > MAX_SENSOR_SYNC_NS:
            return None
        return timestamp, cones

    def _fuse_camera_color(self, camera_cones, lidar_cones, cam_ts, lidar_ts, image_width, speed, yaw_rate):
        """Attach camera colour to a nearest-timestamp LiDAR scan."""
        sync_delta_ns = abs(int(cam_ts) - int(lidar_ts))
        self.metrics["sync_ok"] += 1
        self.metrics["sync_delta_ms"] += sync_delta_ns / 1e6

        matches, used = {}, set()
        focal = image_width / (2.0 * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEGREES) / 2.0))
        dt = (int(cam_ts) - int(lidar_ts)) / 1e9
        rotation, margin = yaw_rate * dt, min(30.0, LIDAR_PROJECTION_MARGIN_PX + focal * abs(yaw_rate * dt))

        for cone in sorted(camera_cones, key=lambda item: item[6], reverse=True):
            center, x1, y1, x2, y2, label, confidence = cone
            best = None
            for index, lidar in enumerate(lidar_cones):
                if index in used:
                    continue
                _, _, _, left, forward = lidar
                vehicle_x, vehicle_y = forward + LIDAR_X_M - speed * dt, left + LIDAR_Y_M
                camera_x = math.cos(rotation) * vehicle_x + math.sin(rotation) * vehicle_y - CAMERA_X_M
                camera_y = -math.sin(rotation) * vehicle_x + math.cos(rotation) * vehicle_y - CAMERA_Y_M
                if camera_x <= 0.0:
                    continue
                projected_x = image_width / 2.0 - focal * camera_y / camera_x
                error = abs(projected_x - center)
                if x1 - margin <= projected_x <= x2 + margin and (best is None or error < best[0]):
                    best = (error, index)
            if best:
                error, index = best
                used.add(index)
                matches[index] = (label, confidence, [int(x1), int(y1), int(x2), int(y2)], error, margin)

        observations = []
        for index, (label, confidence, box, error, match_margin) in matches.items():
            px, py, _, left, forward = lidar_cones[index]
            vehicle_x, vehicle_y = forward + LIDAR_X_M, left + LIDAR_Y_M
            distance = math.hypot(vehicle_x, vehicle_y)
            if not 0.5 <= distance <= COLOR_MAX_RANGE_M:
                continue
            observations.append({
                "label": label, "color": label, "conf": confidence,
                "cam_box": box, "box_height": float(box[3] - box[1]) if box else 0.0,
                "lidar_pixel": [int(px), int(py)], "range": distance,
                "bearing": math.atan2(vehicle_y, vehicle_x),
                "lidar_timestamp_ns": int(lidar_ts), "sync_delta_ns": sync_delta_ns,
                "projection_error_px": error, "projection_margin_px": match_margin,
            })
        self.metrics["matches"] += len(matches)
        return observations

    @staticmethod
    def _normalize_cone_color(label):
        value = str(label).lower()
        for color in ("blue", "yellow", "orange"):
            if color in value:
                return color
        return None

    def _mapping_measurements(self, measurements):
        """Associate timestamp-corrected geometry and accumulate colour evidence."""
        if not self._mapping_is_healthy():
            # A bad pose turns stable LiDAR cones into random world points.
            # Do not move, create, or promote landmarks until localization is
            # healthy again; delayed YOLO colours are handled separately.
            self.metrics["mapping_frozen"] += len(measurements)
            return []
        self.candidate_scan_index += 1
        # Once the first lap has produced a stable map, register every new
        # LiDAR scan to those fixed cones *before* mapping it.  Previously the
        # scan was written in the drifting IMU frame first, which made the
        # second lap a second, offset track even when orange cones were later
        # recognised.
        self._localize_scan_to_persistent_map(measurements)
        used_candidates = set()
        mapping_measurements = []
        current_x, current_y, current_theta = self.ekf.x[:3]
        self._refresh_active_ekf_window(current_x, current_y)

        for measurement in sorted(measurements, key=lambda item: item["range"]):
            r = float(measurement["range"])
            b = float(measurement["bearing"])
            pose, latency_s = self._pose_at(int(measurement.get("lidar_timestamp_ns", 0)))
            pose_x, pose_y, pose_theta, pose_speed = pose
            global_x = pose_x + r * math.cos(pose_theta + b)
            global_y = pose_y + r * math.sin(pose_theta + b)

            pose_sigma = math.sqrt(max(0.0, float(self.ekf.P[0, 0] + self.ekf.P[1, 1])))
            association_gate = min(
                CANDIDATE_MAX_GATE_M,
                CANDIDATE_BASE_GATE_M + pose_speed * min(latency_s, 0.2) + 2.0 * pose_sigma,
            )

            best_candidate = None
            best_distance = association_gate
            for candidate in self._nearby_candidates(global_x, global_y, association_gate):
                if candidate["id"] in used_candidates:
                    continue
                distance = math.hypot(
                    candidate["x"] - global_x,
                    candidate["y"] - global_y,
                )
                if distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate

            if best_candidate is None:
                if getattr(self, "persistent_map_locked", False):
                    # After a verified lap closure, the learned map is the
                    # source of truth.  An unmatched return observation is a
                    # localization outlier, not a new cone on a second lap.
                    self.metrics["repeat_lap_rejects"] += 1
                    continue
                if len(self.cone_candidates) >= CANDIDATE_GLOBAL_MAP_CAPACITY:
                    self.metrics["map_capacity_rejects"] += 1
                    continue
                position_weight = 1.0 / (0.10 + 0.02 * r) ** 2
                candidate = {
                    "id": self.next_candidate_id,
                    "x": global_x,
                    "y": global_y,
                    "position_hits": 1,
                    "position_weight": position_weight,
                    "position_error_sq_sum": 0.0,
                    "position_uncertainty": float("inf"),
                    "last_seen_scan": self.candidate_scan_index,
                    "created_scan": self.candidate_scan_index,
                    "color_scores": {"blue": 0.0, "yellow": 0.0, "orange": 0.0},
                    "color": None,
                    "color_source": "unknown",
                    "geometry_confirmed": False,
                    "color_confirmed": False,
                }
                self.next_candidate_id += 1
                self.cone_candidates.append(candidate)
                self._index_candidate(candidate)
                self.metrics["candidate_new"] += 1
            else:
                candidate = best_candidate
                residual = math.hypot(candidate["x"] - global_x, candidate["y"] - global_y)
                candidate["position_hits"] += 1
                measurement_weight = 1.0 / (0.10 + 0.02 * r) ** 2
                total_weight = candidate["position_weight"] + measurement_weight
                candidate["x"] = (
                    candidate["x"] * candidate["position_weight"]
                    + global_x * measurement_weight
                ) / total_weight
                candidate["y"] = (
                    candidate["y"] * candidate["position_weight"]
                    + global_y * measurement_weight
                ) / total_weight
                candidate["position_weight"] = total_weight
                candidate["position_error_sq_sum"] += residual * residual
                candidate["position_uncertainty"] = math.sqrt(
                    candidate["position_error_sq_sum"]
                    / max(1, candidate["position_hits"] - 1)
                )
                candidate["last_seen_scan"] = self.candidate_scan_index
                self._index_candidate(candidate)
                self.metrics["candidate_matches"] += 1

            used_candidates.add(candidate["id"])
            if (
                not candidate["geometry_confirmed"]
                and candidate["position_hits"] >= MIN_GEOMETRY_CONFIRMATIONS
                and candidate["position_uncertainty"] <= min(
                    association_gate, CANDIDATE_MAX_POSITION_UNCERTAINTY_M
                )
            ):
                candidate["geometry_confirmed"] = True
                self.last_geometry_promotion_scan = self.candidate_scan_index
                self.metrics["geometry_confirmed"] += 1

            if (
                candidate["geometry_confirmed"]
                and self.ekf.can_track_candidate(candidate["id"])
            ):
                accepted = dict(measurement)
                accepted["color"] = (
                    candidate["color"] if candidate["color_confirmed"] else "unknown"
                )
                accepted["candidate_id"] = candidate["id"]
                # EKF expects a current-pose relative observation. The global
                # point above was calculated from the historical capture pose.
                dx, dy = global_x - current_x, global_y - current_y
                accepted["range"] = math.hypot(dx, dy)
                accepted["bearing"] = self.ekf.normalize_angle(
                    math.atan2(dy, dx) - current_theta
                )
                mapping_measurements.append(accepted)

        expired = [
            candidate for candidate in self.cone_candidates
            if not candidate["geometry_confirmed"]
            and self.candidate_scan_index - candidate["last_seen_scan"] > CANDIDATE_EXPIRY_SCANS
        ]
        if expired:
            expired_ids = {candidate["id"] for candidate in expired}
            self.cone_candidates = [
                candidate for candidate in self.cone_candidates
                if candidate["id"] not in expired_ids
            ]
            for candidate in expired:
                self.candidate_by_id.pop(candidate["id"], None)
                self.candidate_spatial_index[candidate.get("_grid_cell")].discard(candidate["id"])
        return mapping_measurements

    def _localize_scan_to_persistent_map(self, measurements):
        """Apply a small, robust scan-to-map pose correction before map writes."""
        if not getattr(self, "initial_map_ready", False) or len(measurements) < MAP_LOCALIZATION_MIN_MATCHES:
            return

        source_points = []
        for measurement in measurements:
            try:
                r = float(measurement["range"])
                bearing = float(measurement["bearing"])
            except (KeyError, TypeError, ValueError):
                continue
            pose, _ = self._pose_at(int(measurement.get("lidar_timestamp_ns", 0)))
            source_points.append((
                pose[0] + r * math.cos(pose[2] + bearing),
                pose[1] + r * math.sin(pose[2] + bearing),
            ))

        matches, used_ids = [], set()
        minimum_reference_scan = self.candidate_scan_index - MAP_REFERENCE_MIN_AGE_SCANS
        references = [
            candidate for candidate in self.cone_candidates
            if candidate.get("geometry_confirmed")
            and candidate.get("created_scan", -MAP_REFERENCE_MIN_AGE_SCANS) <= minimum_reference_scan
        ]
        if len(references) < MAP_LOCALIZATION_MIN_MATCHES:
            return
        for x, y in source_points:
            reference = min(
                (candidate for candidate in references if candidate["id"] not in used_ids),
                key=lambda candidate: math.hypot(candidate["x"] - x, candidate["y"] - y),
                default=None,
            )
            if reference and math.hypot(reference["x"] - x, reference["y"] - y) <= MAP_LOCALIZATION_GATE_M:
                matches.append(((x, y), reference))
                used_ids.add(reference["id"])
        if len(matches) < MAP_LOCALIZATION_MIN_MATCHES:
            return

        source = np.asarray([point for point, _ in matches], dtype=np.float64)
        target = np.asarray([(candidate["x"], candidate["y"]) for _, candidate in matches], dtype=np.float64)
        rotation, translation = self._rigid_registration(source, target)
        corrected = (rotation @ source.T).T + translation
        residual = float(np.sqrt(np.mean(np.sum((corrected - target) ** 2, axis=1))))
        yaw_correction = math.atan2(rotation[1, 0], rotation[0, 0])
        if (
            residual > MAP_LOCALIZATION_MAX_RESIDUAL_M
            or float(np.linalg.norm(translation)) > MAP_LOCALIZATION_MAX_TRANSLATION_M
            or abs(yaw_correction) > MAP_LOCALIZATION_MAX_YAW_RAD
        ):
            self.metrics["map_localization_rejects"] += 1
            return
        self._transform_ekf_frame(rotation, translation, yaw_correction)
        self.metrics["map_localizations"] += 1

    @staticmethod
    def _rigid_registration(source, target):
        """Return the best 2-D rotation and translation from source to target."""
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1, :] *= -1.0
            rotation = vt.T @ u.T
        return rotation, target_center - rotation @ source_center

    def _refresh_active_ekf_window(self, car_x, car_y):
        """Keep the dense EKF local while the global BEV map stays complete."""
        nearby = sorted(
            (
                candidate for candidate in self.cone_candidates
                if candidate.get("geometry_confirmed")
                and math.hypot(candidate["x"] - car_x, candidate["y"] - car_y)
                <= CANDIDATE_EKF_WINDOW_RADIUS_M
            ),
            key=lambda candidate: math.hypot(
                candidate["x"] - car_x, candidate["y"] - car_y
            ),
        )[:MAX_ACTIVE_EKF_LANDMARKS]
        self.ekf.retain_candidate_ids({candidate["id"] for candidate in nearby})

    def _mapping_is_healthy(self):
        """Reject map writes while vehicle motion or pose uncertainty is unsafe."""
        _, yaw_rate = self._current_motion()
        position_sigma = math.sqrt(max(0.0, float(self.ekf.P[0, 0] + self.ekf.P[1, 1])))
        return (
            abs(yaw_rate) <= MAX_MAPPING_YAW_RATE_RAD_S
            and position_sigma <= MAX_MAPPING_POSITION_SIGMA_M
        )

    @staticmethod
    def _candidate_cell(x, y):
        return (
            math.floor(float(x) / CONE_GRID_CELL_METERS),
            math.floor(float(y) / CONE_GRID_CELL_METERS),
        )

    def _rebuild_candidate_spatial_index(self):
        self.candidate_by_id = {}
        self.candidate_spatial_index = defaultdict(set)
        for candidate in self.cone_candidates:
            self._index_candidate(candidate)

    def _index_candidate(self, candidate):
        """Refresh an updated cone's grid entry without rebuilding the map."""
        if not hasattr(self, "candidate_spatial_index"):
            self._rebuild_candidate_spatial_index()
        previous_cell = candidate.get("_grid_cell")
        cell = self._candidate_cell(candidate["x"], candidate["y"])
        if previous_cell is not None and previous_cell != cell:
            self.candidate_spatial_index[previous_cell].discard(candidate["id"])
        candidate["_grid_cell"] = cell
        self.candidate_by_id[candidate["id"]] = candidate
        self.candidate_spatial_index[cell].add(candidate["id"])

    def _nearby_candidates(self, x, y, radius):
        if not hasattr(self, "candidate_spatial_index"):
            self._rebuild_candidate_spatial_index()
        cell_x, cell_y = self._candidate_cell(x, y)
        cells = max(1, math.ceil(radius / CONE_GRID_CELL_METERS))
        candidates = []
        for ix in range(cell_x - cells, cell_x + cells + 1):
            for iy in range(cell_y - cells, cell_y + cells + 1):
                for identifier in self.candidate_spatial_index.get((ix, iy), ()):
                    candidate = self.candidate_by_id.get(identifier)
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    def _apply_color_measurements(self, measurements):
        """Update semantics only; delayed YOLO must not duplicate geometry."""
        orange_observations = []
        for measurement in measurements:
            r, bearing = float(measurement["range"]), float(measurement["bearing"])
            pose, _ = self._pose_at(int(measurement["lidar_timestamp_ns"]))
            x = pose[0] + r * math.cos(pose[2] + bearing)
            y = pose[1] + r * math.sin(pose[2] + bearing)
            candidate = min(
                self._nearby_candidates(x, y, CANDIDATE_MAX_GATE_M),
                key=lambda item: math.hypot(item["x"] - x, item["y"] - y),
                default=None,
            )
            if candidate and math.hypot(candidate["x"] - x, candidate["y"] - y) <= CANDIDATE_MAX_GATE_M:
                self._accumulate_color(candidate, measurement)
            if self._normalize_cone_color(measurement.get("color")) == "orange":
                orange_observations.append((x, y))

        self._apply_orange_loop_closure(orange_observations)
        self._capture_orange_references()

    def _capture_orange_references(self):
        """Freeze the first coloured orange gate as the global lap anchor."""
        if not hasattr(self, "orange_reference_ids"):
            self.orange_reference_ids = set()
        if self.orange_reference_ids:
            return
        anchors = [
            candidate for candidate in self.cone_candidates
            if candidate.get("geometry_confirmed")
            and candidate.get("color") == "orange"
        ]
        # A single orange detection is ambiguous.  A gate needs at least two
        # fixed points before it may correct the global frame.
        if len(anchors) >= 2:
            self.orange_reference_ids = {candidate["id"] for candidate in anchors}

    def _apply_orange_loop_closure(self, observations):
        """Register a returning orange gate to its first-lap map positions.

        LiDAR geometry is intentionally allowed before YOLO labels arrive.
        On a later lap that means a drifting pose can briefly create unknown
        duplicates.  A rigid registration of two or more orange observations
        corrects the EKF frame, then merges those transient duplicates into
        the original persistent cones.
        """
        if len(observations) < 2 or not getattr(self, "orange_reference_ids", set()):
            return
        references = [
            self.candidate_by_id.get(identifier)
            for identifier in self.orange_reference_ids
        ]
        references = [candidate for candidate in references if candidate is not None]
        if len(references) < 2:
            return

        matches, used_ids = [], set()
        for x, y in observations:
            options = [
                candidate for candidate in references
                if candidate["id"] not in used_ids
            ]
            if not options:
                break
            reference = min(
                options,
                key=lambda candidate: math.hypot(candidate["x"] - x, candidate["y"] - y),
            )
            if math.hypot(reference["x"] - x, reference["y"] - y) <= ORANGE_LOOP_CLOSURE_GATE_M:
                matches.append(((x, y), reference))
                used_ids.add(reference["id"])
        if len(matches) < 2:
            return

        source = np.asarray([point for point, _ in matches], dtype=np.float64)
        target = np.asarray([(candidate["x"], candidate["y"]) for _, candidate in matches], dtype=np.float64)
        rotation, translation = self._rigid_registration(source, target)
        yaw_correction = math.atan2(rotation[1, 0], rotation[0, 0])
        if (
            float(np.linalg.norm(translation)) > ORANGE_LOOP_CLOSURE_MAX_TRANSLATION_M
            or abs(yaw_correction) > ORANGE_LOOP_CLOSURE_MAX_YAW_RAD
        ):
            self.metrics["orange_loop_rejects"] += 1
            return

        self._transform_ekf_frame(rotation, translation, yaw_correction)
        self._merge_current_scan_duplicates(rotation, translation)
        self.persistent_map_locked = True
        self.metrics["orange_loop_closures"] += 1

    def _transform_ekf_frame(self, rotation, translation, yaw_correction):
        """Move the local EKF working set into the fixed first-lap frame."""
        for index in range(0, len(self.ekf.x), 2):
            if index == 2:
                continue
            self.ekf.x[index:index + 2] = rotation @ self.ekf.x[index:index + 2] + translation
        self.ekf.x[2] = self.ekf.normalize_angle(self.ekf.x[2] + yaw_correction)
        corrected_history = deque(maxlen=self.pose_history.maxlen)
        for timestamp, x, y, heading, speed, yaw_rate in self.pose_history:
            x, y = rotation @ np.asarray([x, y]) + translation
            corrected_history.append((
                timestamp, float(x), float(y),
                self.ekf.normalize_angle(heading + yaw_correction), speed, yaw_rate,
            ))
        self.pose_history = corrected_history

    def _merge_current_scan_duplicates(self, rotation, translation):
        """Replace post-drift candidates with their first-lap neighbours."""
        fresh = [
            candidate for candidate in self.cone_candidates
            if candidate.get("created_scan", -ORANGE_LOOP_CLOSURE_RECENT_SCANS)
            >= self.candidate_scan_index - ORANGE_LOOP_CLOSURE_RECENT_SCANS
            and candidate["id"] not in getattr(self, "orange_reference_ids", set())
        ]
        if not fresh:
            return
        remove_ids = set()
        for candidate in fresh:
            x, y = rotation @ np.asarray([candidate["x"], candidate["y"]]) + translation
            candidate["x"], candidate["y"] = float(x), float(y)
            neighbours = [
                other for other in self.cone_candidates
                if other["id"] != candidate["id"] and other["id"] not in remove_ids
                and other.get("created_scan", -ORANGE_LOOP_CLOSURE_RECENT_SCANS)
                < self.candidate_scan_index - ORANGE_LOOP_CLOSURE_RECENT_SCANS
            ]
            existing = min(
                neighbours,
                key=lambda other: math.hypot(other["x"] - candidate["x"], other["y"] - candidate["y"]),
                default=None,
            )
            if existing and math.hypot(existing["x"] - candidate["x"], existing["y"] - candidate["y"]) <= CANDIDATE_MAX_GATE_M:
                existing["position_hits"] += candidate["position_hits"]
                existing["last_seen_scan"] = self.candidate_scan_index
                remove_ids.add(candidate["id"])
        if remove_ids:
            self.cone_candidates = [
                candidate for candidate in self.cone_candidates
                if candidate["id"] not in remove_ids
            ]
        self._rebuild_candidate_spatial_index()

    def _accumulate_color(self, candidate, measurement):
        color, weight = self._normalize_cone_color(measurement.get("color")), self._color_weight(measurement)
        if not color or weight <= 0.0:
            self.metrics["color_rejects"] += int(color is not None)
            return
        scores = candidate["color_scores"]
        for name in scores:
            scores[name] *= 0.98
        scores[color] += weight
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        winner, winner_score = ordered[0]
        total_score = sum(scores.values())
        if (
            winner_score >= COLOR_CONFIRM_SCORE
            and winner_score - ordered[1][1] >= COLOR_CONFIRM_MARGIN
            and winner_score / total_score >= MIN_COLOR_WINNER_SHARE
        ):
            if candidate.get("color_source") != "vision":
                self.metrics["color_confirmed"] += 1
            self._set_candidate_color(candidate, winner, "vision")
        self.metrics["color_weight"] += weight

    def _set_candidate_color(self, candidate, color, source):
        candidate["color_confirmed"] = True
        candidate["color"], candidate["color_source"] = color, source
        for landmark in self.ekf.landmarks:
            if landmark.get("candidate_id") == candidate["id"]:
                landmark["color"] = color
                break

    def _infer_missing_colors(self):
        """Fill passed boundary gaps; protect orange four-cone gates."""
        car_x, car_y, heading = self.ekf.x[:3]
        # Only nearby cones can affect the next path. Historical cones stay in
        # the global map but are excluded from the expensive colour inference.
        confirmed = [
            c for c in self.cone_candidates
            if c["geometry_confirmed"]
            and math.hypot(c["x"] - car_x, c["y"] - car_y) <= COLOR_INFERENCE_RADIUS_M
        ]
        orange_groups = set()
        for seed in confirmed:
            if seed["color_scores"]["orange"] < 0.1:
                continue
            group = sorted(
                confirmed,
                key=lambda item: math.hypot(item["x"] - seed["x"], item["y"] - seed["y"]),
            )[:4]
            if (
                len(group) == 4
                and max(math.hypot(c["x"] - seed["x"], c["y"] - seed["y"]) for c in group) <= ORANGE_GROUP_RANGE_M
                and sum(c["color_scores"]["orange"] >= 0.1 for c in group) >= 2
                and sum(c["color_scores"]["orange"] for c in group) >= 0.5
                and not any(c.get("color_source") == "vision" and c.get("color") != "orange" for c in group)
                and self._is_gate_group(group)
            ):
                orange_groups.update(c["id"] for c in group)
                for candidate in group:
                    if candidate.get("color_source") != "vision":
                        self._set_candidate_color(candidate, "orange", "orange_group")

        for candidate in confirmed:
            behind = math.cos(heading) * (candidate["x"] - car_x) + math.sin(heading) * (candidate["y"] - car_y)
            if (
                behind >= -1.0
                or candidate.get("color_source") in ("vision", "orange_group", "boundary")
                or candidate["id"] in orange_groups
                or candidate["color_scores"]["orange"] >= 0.1
            ):
                continue

            choices = []
            for color in ("blue", "yellow"):
                neighbours = sorted(
                    (c for c in confirmed if c.get("color") == color
                     and c.get("color_source") == "vision" and c["id"] != candidate["id"]),
                    key=lambda c: math.hypot(c["x"] - candidate["x"], c["y"] - candidate["y"]),
                )[:2]
                if len(neighbours) < 2:
                    continue
                distances = [math.hypot(c["x"] - candidate["x"], c["y"] - candidate["y"]) for c in neighbours]
                error = self._segment_error(candidate, *neighbours)
                if max(distances) <= BOUNDARY_NEIGHBOR_RANGE_M and error <= BOUNDARY_MAX_ERROR_M:
                    choices.append((error + 0.05 * sum(distances), color))

            choices.sort()
            if choices and (len(choices) == 1 or choices[1][0] - choices[0][0] >= 0.3):
                self._set_candidate_color(candidate, choices[0][1], "boundary")
                self.metrics["color_inferred"] += 1

    @staticmethod
    def _segment_error(point, first, second):
        vx, vy = second["x"] - first["x"], second["y"] - first["y"]
        length_sq = vx * vx + vy * vy
        if length_sq < 1e-6:
            return float("inf")
        t = ((point["x"] - first["x"]) * vx + (point["y"] - first["y"]) * vy) / length_sq
        if not 0.0 < t < 1.0:
            return float("inf")
        projection_x, projection_y = first["x"] + t * vx, first["y"] + t * vy
        return math.hypot(point["x"] - projection_x, point["y"] - projection_y)

    @staticmethod
    def _is_gate_group(group):
        points = np.array([(c["x"], c["y"]) for c in group], dtype=np.float64)
        values, vectors = np.linalg.eigh(np.cov((points - points.mean(axis=0)).T))
        if values[0] < 0.04:
            return False
        projected = (points - points.mean(axis=0)) @ vectors
        for axis in range(2):
            ordered = np.sort(projected[:, axis])
            gap = ordered[2:].mean() - ordered[:2].mean()
            spread = max(ordered[1] - ordered[0], ordered[3] - ordered[2])
            if gap < 0.5 or spread > 0.6 * gap:
                return False
        return True

    def _pose_at(self, timestamp_ns):
        """Interpolate the EKF pose at a simulator sensor timestamp."""
        if not self.pose_history or timestamp_ns <= 0:
            speed, _ = self._current_motion()
            return (float(self.ekf.x[0]), float(self.ekf.x[1]), float(self.ekf.x[2]), speed), 0.0

        history = list(self.pose_history)
        latest_timestamp = history[-1][0]
        timing_offset_s = abs(latest_timestamp - timestamp_ns) / 1e9
        if timestamp_ns <= history[0][0]:
            return history[0][1:5], timing_offset_s
        if timestamp_ns >= latest_timestamp:
            _, x, y, heading, speed, yaw_rate = history[-1]
            dt = min(0.1, (timestamp_ns - latest_timestamp) / 1e9)
            return (
                x + speed * math.cos(heading) * dt,
                y + speed * math.sin(heading) * dt,
                self.ekf.normalize_angle(heading + yaw_rate * dt),
                speed,
            ), timing_offset_s

        for older, newer in zip(history, history[1:]):
            if older[0] <= timestamp_ns <= newer[0]:
                fraction = (timestamp_ns - older[0]) / max(1, newer[0] - older[0])
                heading_delta = self.ekf.normalize_angle(newer[3] - older[3])
                pose = (
                    older[1] + fraction * (newer[1] - older[1]),
                    older[2] + fraction * (newer[2] - older[2]),
                    self.ekf.normalize_angle(older[3] + fraction * heading_delta),
                    older[4] + fraction * (newer[4] - older[4]),
                )
                return pose, timing_offset_s
        return history[-1][1:5], timing_offset_s

    @staticmethod
    def _color_weight(measurement):
        color = TestConsoleApp._normalize_cone_color(measurement.get("color"))
        distance = float(measurement.get("range", 0.0))
        confidence = float(measurement.get("conf", 0.0))
        if color is None or distance > COLOR_MAX_RANGE_M or confidence < YOLO_MIN_CONFIDENCE:
            return 0.0

        box_quality = min(1.0, float(measurement.get("box_height", 0.0)) / 30.0)
        range_quality = max(0.0, 1.0 - distance / COLOR_MAX_RANGE_M)
        sync_quality = max(0.0, 1.0 - float(measurement.get("sync_delta_ns", 0)) / MAX_SENSOR_SYNC_NS)
        margin = max(1.0, float(measurement.get("projection_margin_px", 1.0)))
        projection_quality = max(0.0, 1.0 - float(measurement.get("projection_error_px", margin)) / margin)
        return confidence * (0.35 + 0.65 * range_quality) * (0.4 + 0.6 * box_quality) \
            * (0.5 + 0.5 * sync_quality) * (0.5 + 0.5 * projection_quality)

    def _update_map_readiness(self):
        geometry_count = len(self.ekf.landmarks)
        colors = [landmark.get("color", "unknown").lower() for landmark in self.ekf.landmarks]
        has_blue = any("blue" in color for color in colors)
        has_yellow = any("yellow" in color for color in colors)
        stable = (
            self.candidate_scan_index - self.last_geometry_promotion_scan
            >= INITIAL_MAP_STABLE_SCANS
        )

        if geometry_count < MIN_INITIAL_MAP_LANDMARKS:
            self.map_readiness_reason = (
                f"Need {MIN_INITIAL_MAP_LANDMARKS - geometry_count} more geometry landmarks"
            )
        elif not has_blue or not has_yellow:
            self.map_readiness_reason = "Need at least one blue and one yellow cone"
        elif not stable:
            scans_left = INITIAL_MAP_STABLE_SCANS - (
                self.candidate_scan_index - self.last_geometry_promotion_scan
            )
            self.map_readiness_reason = f"Waiting {max(0, scans_left)} stable scans"
        else:
            self.initial_map_ready = True
            self.map_readiness_reason = "Ready"

    def refresh_ui(self):
        if self.shutdown_event.is_set():
            return

        # Schedule first so an exception in this refresh does not permanently
        # stop keyboard, vision, EKF, and map updates.
        self.root.after(50, self.refresh_ui)

        # Update inputs based on keyboard state
        self._update_keyboard_inputs()

        speed = 0.0
        yaw_rate = 0.0
        qx = qy = qz = 0.0
        qw = 1.0
        receipt_timestamp_ms = sensor_timestamp_ns = 0
        if self.latest_imu:
            receipt_timestamp_ms = int(self.latest_imu.get("timestamp_ms", 0))
            sensor_timestamp_ns = int(self.latest_imu.get("sensor_timestamp_ns", 0))
            speed = self.latest_imu.get("ground_speed_mps", 0.0)
            yaw_rate = self.latest_imu.get("imu", {}).get("angular_velocity", {}).get("z", 0.0)
            ori = self.latest_imu.get("imu", {}).get("orientation", {})
            qx = ori.get("x", 0.0)
            qy = ori.get("y", 0.0)
            qz = ori.get("z", 0.0)
            qw = ori.get("w", 1.0)

        quaternion_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        imu_is_fresh = (
            receipt_timestamp_ms > 0
            and int(time.time() * 1000) - receipt_timestamp_ms <= IMU_FRESHNESS_LIMIT_MS
        )
        new_imu = sensor_timestamp_ns > self.last_imu_sensor_timestamp_ns
        if imu_is_fresh and new_imu and math.isfinite(quaternion_norm) and quaternion_norm > 1e-9:
            dt = 0.0
            if self.last_imu_sensor_timestamp_ns:
                dt = min(0.2, (sensor_timestamp_ns - self.last_imu_sensor_timestamp_ns) / 1e9)
            self.last_imu_sensor_timestamp_ns = sensor_timestamp_ns

            abs_theta = self.ekf.normalize_angle(
                math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
            )
            if not self.ekf.heading_initialized:
                self.ekf.x[2] = abs_theta
                self.ekf.heading_initialized = True
            else:
                if dt > 0.0:
                    self.ekf.predict(speed, yaw_rate, dt)
                self.ekf.update_heading(abs_theta)
            self.pose_history.append((
                sensor_timestamp_ns, float(self.ekf.x[0]), float(self.ekf.x[1]),
                float(self.ekf.x[2]), abs(float(speed)), float(yaw_rate),
            ))
            self.metrics["imu_updates"] += 1
        elif not imu_is_fresh:
            self.metrics["stale_imu"] += 1

        # Drain fast geometry and slower semantic events without blocking the UI.
        geometry_updated = False
        if self.ekf.heading_initialized:
            while True:
                try:
                    kind, measurements = self.mapping_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "geometry":
                    accepted = self._mapping_measurements(measurements)
                    if accepted:
                        self.ekf.update(accepted)
                        self.metrics["ekf_updates"] += len(accepted)
                    geometry_updated = True
                else:
                    self._apply_color_measurements(measurements)
            if geometry_updated:
                self._infer_missing_colors()
                self._update_map_readiness()

        desired = self._get_desired()
        self.throttle_label.set(f"Throttle: {desired['throttle']:.3f}")
        self.brake_label.set(f"Brake: {desired['brake']:.3f}")
        self.steering_label.set(f"Steering: {desired['steering']:.3f}")

        mode_str = "AUTO" if self.autonomous_mode else "MANUAL"
        self.imu_status_var.set(f"81 IMU: {self.imu_status} [{mode_str}]")
        self.act_status_var.set(f"83 Actuator: {self.act_status}")
        self.vision_status_var.set(f"84 Vision: {self.vision_status}")
        self.ctrl_status_var.set(f"82 Control TX: {self.ctrl_status}")

        if self.latest_imu:
            try:
                speed_val = self.latest_imu.get("ground_speed_mps", 0.0)
                imu = self.latest_imu.get("imu", {})

                av = imu.get("angular_velocity", {})
                la = imu.get("linear_acceleration", {})
                ori = imu.get("orientation", {})

                self.speed_var.set(f"Ground Speed: {speed_val:.3f} m/s")
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

        # Fetch latest images and fused data under thread lock
        with self.yolo_lock:
            cam_img = self.latest_cam_img.copy() if self.latest_cam_img is not None else None
            lidar_img = self.latest_lidar_img.copy() if self.latest_lidar_img is not None else None
            geometry = list(self.latest_geometry_measurements)
            colored = list(self.latest_fused_measurements)
            colors_by_pixel = {tuple(item["lidar_pixel"]): item for item in colored}
            fused = [colors_by_pixel.get(tuple(item["lidar_pixel"]), item) for item in geometry]

        if cam_img is not None and lidar_img is not None:
            try:
                # Draw fused results on the dashboard panels
                for cone in fused:
                    label = cone["label"]
                    conf = cone["conf"]
                    cam_box = cone["cam_box"]
                    lidar_pixel = cone["lidar_pixel"]

                    # Color bounding boxes based on the actual class label (Yellow/Blue/Orange)
                    lbl = label.lower()
                    if "yellow" in lbl:
                        box_color = (255, 255, 0)  # Yellow (RGB)
                    elif "blue" in lbl:
                        box_color = (0, 120, 255)  # Blue (RGB)
                    elif "orange" in lbl:
                        box_color = (255, 165, 0)  # Orange (RGB)
                    else:
                        box_color = (255, 255, 255) # White (RGB)

                    # White is LiDAR-only geometry; green also has camera colour.
                    if lidar_pixel is not None:
                        r_text = f" [{cone['range']:.1f}m]"
                        circle_color = (0, 255, 0) if cam_box else (255, 255, 255)
                        cv2.circle(lidar_img, tuple(lidar_pixel), 10, circle_color, 2)
                    else:
                        r_text = ""

                    if cam_box:
                        x1, y1, x2, y2 = cam_box
                        cv2.rectangle(cam_img, (x1, y1), (x2, y2), box_color, 2)
                        text = f"{label} {conf:.2f}{r_text}"
                        cv2.putText(cam_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Keep both live views independent: camera right, LiDAR left.
                self._display_sensor_image(
                    self.camera_label, cam_img, (520, 390),
                )
                self._display_sensor_image(
                    self.lidar_label, lidar_img, (420, 300),
                )
            except Exception as e:
                self.vision_status = f"Dashboard rendering error ({e})"

        # Update EKF Measurements Text Panel
        if fused:
            lines = []
            for cone in fused[:5]:
                r_val = f"{cone['range']:.2f}m" if cone['range'] is not None else "---"
                b_deg = math.degrees(cone['bearing'])
                b_val = f"{b_deg:+.1f}°"
                lines.append(f"{cone['label'][:3].upper()}: r={r_val:<6} b={b_val}")
            if len(fused) > 5:
                lines.append(f"... and {len(fused) - 5} more")
            self.slam_text_var.set("\n".join(lines))
        else:
            self.slam_text_var.set("No detections")

    def _world_to_map_pixel(self, world_x, world_y):
        usable_width = MAP_WIDTH_PX - 2 * MAP_MARGIN_PX
        usable_height = MAP_HEIGHT_PX - 2 * MAP_MARGIN_PX
        px = MAP_MARGIN_PX + (
            (world_x - self.map_x_min_m) / (self.map_x_max_m - self.map_x_min_m)
        ) * usable_width
        py = MAP_HEIGHT_PX - MAP_MARGIN_PX - (
            (world_y - self.map_y_min_m) / (self.map_y_max_m - self.map_y_min_m)
        ) * usable_height
        return int(round(px)), int(round(py))

    @staticmethod
    def _nice_grid_spacing(span_m):
        """Choose a readable 1/2/5 x 10^n grid interval."""
        target = max(1.0, float(span_m) / 8.0)
        exponent = math.floor(math.log10(target))
        fraction = target / (10.0 ** exponent)
        if fraction <= 1.0:
            nice_fraction = 1.0
        elif fraction <= 2.0:
            nice_fraction = 2.0
        elif fraction <= 5.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
        return nice_fraction * (10.0 ** exponent)

    def _update_map_bounds(self):
        """Expand the world viewport to include the car and EKF landmarks."""
        points = [(float(self.ekf.x[0]), float(self.ekf.x[1]))]
        for candidate in self.cone_candidates:
            if candidate.get("geometry_confirmed"):
                point = (float(candidate["x"]), float(candidate["y"]))
                if all(math.isfinite(value) for value in point):
                    points.append(point)

        finite_points = [
            point for point in points if all(math.isfinite(value) for value in point)
        ]
        if not finite_points:
            return

        wanted_x_min = min(point[0] for point in finite_points) - MAP_EDGE_PADDING_M
        wanted_x_max = max(point[0] for point in finite_points) + MAP_EDGE_PADDING_M
        wanted_y_min = min(point[1] for point in finite_points) - MAP_EDGE_PADDING_M
        wanted_y_max = max(point[1] for point in finite_points) + MAP_EDGE_PADDING_M

        current_span = max(
            self.map_x_max_m - self.map_x_min_m,
            self.map_y_max_m - self.map_y_min_m,
        )
        expansion_step = max(MAP_EDGE_PADDING_M, current_span * MAP_EXPANSION_FRACTION)

        if wanted_x_min < self.map_x_min_m:
            self.map_x_min_m = min(wanted_x_min, self.map_x_min_m - expansion_step)
        if wanted_x_max > self.map_x_max_m:
            self.map_x_max_m = max(wanted_x_max, self.map_x_max_m + expansion_step)
        if wanted_y_min < self.map_y_min_m:
            self.map_y_min_m = min(wanted_y_min, self.map_y_min_m - expansion_step)
        if wanted_y_max > self.map_y_max_m:
            self.map_y_max_m = max(wanted_y_max, self.map_y_max_m + expansion_step)

        # The plot is square, so equalize the world spans. This prevents visual
        # distortion: one metre in X always occupies the same pixels as one in Y.
        x_span = self.map_x_max_m - self.map_x_min_m
        y_span = self.map_y_max_m - self.map_y_min_m
        if x_span < y_span:
            extra = y_span - x_span
            self.map_x_min_m -= extra / 2.0
            self.map_x_max_m += extra / 2.0
        elif y_span < x_span:
            extra = x_span - y_span
            self.map_y_min_m -= extra / 2.0
            self.map_y_max_m += extra / 2.0

    @staticmethod
    def _pixel_is_on_map(pixel):
        px, py = pixel
        return (
            MAP_MARGIN_PX <= px <= MAP_WIDTH_PX - MAP_MARGIN_PX
            and MAP_MARGIN_PX <= py <= MAP_HEIGHT_PX - MAP_MARGIN_PX
        )

    def _draw_ekf_map(self):
        self._update_map_bounds()
        map_img = np.full((MAP_HEIGHT_PX, MAP_WIDTH_PX, 3), 16, dtype=np.uint8)
        plot_min = (MAP_MARGIN_PX, MAP_MARGIN_PX)
        plot_max = (MAP_WIDTH_PX - MAP_MARGIN_PX, MAP_HEIGHT_PX - MAP_MARGIN_PX)
        cv2.rectangle(map_img, plot_min, plot_max, (110, 110, 110), 1)

        map_span_m = self.map_x_max_m - self.map_x_min_m
        grid_spacing_m = self._nice_grid_spacing(map_span_m)
        first_grid_x = math.ceil(self.map_x_min_m / grid_spacing_m) * grid_spacing_m
        first_grid_y = math.ceil(self.map_y_min_m / grid_spacing_m) * grid_spacing_m

        # Global world grid: coordinates do not follow or rotate with the car.
        for grid_x in np.arange(first_grid_x, self.map_x_max_m + 0.1, grid_spacing_m):
            px, _ = self._world_to_map_pixel(grid_x, 0.0)
            axis = abs(grid_x) < 1e-9
            cv2.line(
                map_img,
                (px, MAP_MARGIN_PX),
                (px, MAP_HEIGHT_PX - MAP_MARGIN_PX),
                (105, 105, 105) if axis else (42, 42, 42),
                2 if axis else 1,
            )
            cv2.putText(
                map_img, f"{int(grid_x)}", (px - 9, MAP_HEIGHT_PX - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA,
            )

        for grid_y in np.arange(first_grid_y, self.map_y_max_m + 0.1, grid_spacing_m):
            _, py = self._world_to_map_pixel(0.0, grid_y)
            axis = abs(grid_y) < 1e-9
            cv2.line(
                map_img,
                (MAP_MARGIN_PX, py),
                (MAP_WIDTH_PX - MAP_MARGIN_PX, py),
                (105, 105, 105) if axis else (42, 42, 42),
                2 if axis else 1,
            )
            cv2.putText(
                map_img, f"{int(grid_y)}", (8, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA,
            )

        cv2.putText(map_img, "+X (m)", (MAP_WIDTH_PX - 90, MAP_HEIGHT_PX - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(map_img, "+Y (m)", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 190, 190), 1, cv2.LINE_AA)

        # Fixed-world trajectory.
        trajectory_pixels = [
            self._world_to_map_pixel(tx, ty) for tx, ty in self.ekf.trajectory
        ]
        for start, end in zip(trajectory_pixels, trajectory_pixels[1:]):
            if self._pixel_is_on_map(start) and self._pixel_is_on_map(end):
                cv2.line(map_img, start, end, (0, 180, 255), 2)

        # Controller overlays remain available when autonomy is toggled later.
        if hasattr(self, "controller") and self.controller.last_waypoints:
            waypoint_pixels = [
                self._world_to_map_pixel(wx, wy)
                for wx, wy in self.controller.last_waypoints
            ]
            for pixel in waypoint_pixels:
                if self._pixel_is_on_map(pixel):
                    cv2.circle(map_img, pixel, 3, (255, 0, 255), -1)
            for start, end in zip(waypoint_pixels, waypoint_pixels[1:]):
                if self._pixel_is_on_map(start) and self._pixel_is_on_map(end):
                    cv2.line(map_img, start, end, (255, 0, 255), 1)

        if hasattr(self, "controller") and self.controller.last_target_point:
            target_pixel = self._world_to_map_pixel(*self.controller.last_target_point)
            if self._pixel_is_on_map(target_pixel):
                cv2.circle(map_img, target_pixel, 6, (255, 60, 60), 2)

        # The complete learned centreline is the durable track shape.  It
        # grows continuously, while ``last_waypoints`` remains the current
        # short-horizon purple route used for control.
        track_shape = getattr(getattr(self, "controller", None), "track_shape", [])
        track_pixels = [self._world_to_map_pixel(x, y) for x, y in track_shape]
        for start, end in zip(track_pixels, track_pixels[1:]):
            if self._pixel_is_on_map(start) and self._pixel_is_on_map(end):
                cv2.line(map_img, start, end, (170, 80, 255), 2)
        if getattr(getattr(self, "controller", None), "track_shape_closed", False) and len(track_pixels) > 2:
            cv2.line(map_img, track_pixels[-1], track_pixels[0], (170, 80, 255), 2)

        # Provisional geometry is visible immediately but is not yet in EKF.
        provisional_count = 0
        for candidate in self.cone_candidates:
            if candidate.get("geometry_confirmed", False):
                continue
            pixel = self._world_to_map_pixel(candidate["x"], candidate["y"])
            if not self._pixel_is_on_map(pixel):
                continue
            provisional_count += 1
            cv2.circle(map_img, pixel, 5, (255, 255, 255), 1)
            uncertainty = candidate.get("position_uncertainty", float("inf"))
            if math.isfinite(uncertainty):
                pixels_per_meter = (
                    (MAP_WIDTH_PX - 2 * MAP_MARGIN_PX)
                    / (self.map_x_max_m - self.map_x_min_m)
                )
                radius = max(6, min(24, int(uncertainty * pixels_per_meter)))
                cv2.circle(map_img, pixel, radius, (100, 100, 100), 1)

        unknown_count = 0
        colored_count = 0
        for candidate in self.cone_candidates:
            if not candidate.get("geometry_confirmed"):
                continue
            pixel = self._world_to_map_pixel(candidate["x"], candidate["y"])
            if not self._pixel_is_on_map(pixel):
                continue
            color = str(candidate.get("color") or "unknown").lower()
            if "yellow" in color:
                draw_color = (255, 255, 0)
                colored_count += 1
            elif "blue" in color:
                draw_color = (0, 120, 255)
                colored_count += 1
            elif "orange" in color:
                draw_color = (255, 165, 0)
                colored_count += 1
            else:
                draw_color = (255, 255, 255)
                unknown_count += 1
            cv2.circle(map_img, pixel, 6, draw_color, -1)
            cv2.circle(map_img, pixel, 7, (0, 0, 0), 1)

        # Vehicle moves through the fixed coordinate plane.
        car_x, car_y, theta = self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]
        car_pixel = self._world_to_map_pixel(car_x, car_y)
        if self._pixel_is_on_map(car_pixel):
            cx, cy = car_pixel
            forward = (math.cos(theta), -math.sin(theta))
            screen_left = (-math.sin(theta), -math.cos(theta))
            tip = (int(cx + 13 * forward[0]), int(cy + 13 * forward[1]))
            back_x = cx - 7 * forward[0]
            back_y = cy - 7 * forward[1]
            left = (int(back_x + 8 * screen_left[0]), int(back_y + 8 * screen_left[1]))
            right = (int(back_x - 8 * screen_left[0]), int(back_y - 8 * screen_left[1]))
            triangle = np.array([tip, left, right], dtype=np.int32)
            cv2.drawContours(map_img, [triangle], 0, (0, 255, 0), -1)
            cv2.drawContours(map_img, [triangle], 0, (0, 0, 0), 1)

        mode = "AUTONOMOUS" if self.autonomous_mode else "MANUAL"
        status_lines = [
            f"Car: X={car_x:+.2f}m  Y={car_y:+.2f}m  heading={math.degrees(theta):+.1f}deg",
            f"Cones: provisional={provisional_count}  unknown={unknown_count}  coloured={colored_count}",
            f"Map: {self.map_readiness_reason}  Mode: {mode}",
            f"View: {map_span_m:.0f}m square  Grid: {grid_spacing_m:g}m",
        ]
        for row, text_line in enumerate(status_lines):
            cv2.putText(
                map_img, text_line, (MAP_MARGIN_PX + 8, MAP_MARGIN_PX + 18 + row * 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1, cv2.LINE_AA,
            )
        return map_img

    def _on_key_press(self, event):
        key = event.keysym.lower()
        self.pressed_keys[key] = time.time()
        
        if key == 'p':
            self.toggle_autonomy()

    def _on_key_release(self, event):
        key = event.keysym.lower()
        release_time = time.time()
        self.root.after(20, lambda: self._confirm_release(key, release_time))

    def _confirm_release(self, key, release_time):
        if key in self.pressed_keys and self.pressed_keys[key] > release_time:
            return
        if key in self.pressed_keys:
            del self.pressed_keys[key]

    def _update_keyboard_inputs(self):
        drive_keys = {"left", "right", "up", "down", "a", "d", "w", "s", "space"}
        active_drive_keys = any(k in self.pressed_keys for k in drive_keys)

        if active_drive_keys:
            self.using_keyboard = True

        if not self.using_keyboard:
            return

        current_steering = self.steering_var.get()
        current_throttle = self.throttle_var.get()
        current_brake = self.brake_var.get()

        steer_left = "left" in self.pressed_keys or "a" in self.pressed_keys
        steer_right = "right" in self.pressed_keys or "d" in self.pressed_keys
        throttle_up = "up" in self.pressed_keys or "w" in self.pressed_keys
        brake_down = "down" in self.pressed_keys or "s" in self.pressed_keys
        space_brake = "space" in self.pressed_keys

        changed = False

        # Steering accumulation and decay (20Hz loop, dt=50ms)
        if steer_left and not steer_right:
            new_steering = max(-1.0, current_steering - 0.08)
            if new_steering != current_steering:
                current_steering = new_steering
                changed = True
        elif steer_right and not steer_left:
            new_steering = min(1.0, current_steering + 0.08)
            if new_steering != current_steering:
                current_steering = new_steering
                changed = True
        else:
            # Decay steering back to 0.0
            if current_steering > 0.0:
                current_steering = max(0.0, current_steering - 0.1)
                changed = True
            elif current_steering < 0.0:
                current_steering = min(0.0, current_steering + 0.1)
                changed = True

        # Throttle accumulation and decay
        if throttle_up:
            # Limit to 0.40 for safe testing/driving
            new_throttle = min(0.40, current_throttle + 0.05)
            if new_throttle != current_throttle:
                current_throttle = new_throttle
                changed = True
        else:
            if current_throttle > 0.0:
                current_throttle = max(0.0, current_throttle - 0.08)
                changed = True

        # Brake accumulation and decay
        if space_brake:
            if current_brake != 1.0 or current_throttle != 0.0:
                current_brake = 1.0
                current_throttle = 0.0
                changed = True
        elif brake_down:
            new_brake = min(1.0, current_brake + 0.15)
            if new_brake != current_brake or current_throttle != 0.0:
                current_brake = new_brake
                current_throttle = 0.0
                changed = True
        else:
            if current_brake > 0.0:
                current_brake = max(0.0, current_brake - 0.15)
                changed = True

        if changed:
            self.steering_var.set(round(current_steering, 3))
            self.throttle_var.set(round(current_throttle, 3))
            self.brake_var.set(round(current_brake, 3))
            self.on_slider_change()

        # Turn off using_keyboard state if everything is fully zeroed out and inactive
        if not active_drive_keys and round(current_steering, 3) == 0.0 and round(current_throttle, 3) == 0.0 and round(current_brake, 3) == 0.0:
            self.using_keyboard = False

    def on_close(self):
        if self.closing:
            return

        self.closing = True
        self.autonomous_mode = False
        self.using_keyboard = False
        self._set_desired({"throttle": 0.0, "brake": 1.0, "steering": 0.0})
        self.shutdown_event.set()

        # Give the writer thread time to publish its final full-brake packet.
        self.root.after(100, self.root.destroy)


if __name__ == "__main__":
    root = tk.Tk()
    app = TestConsoleApp(root)
    root.mainloop()
