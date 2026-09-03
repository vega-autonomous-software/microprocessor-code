
this repo has the most stable version of the code in the microprosessor 




# Autonomous Car Project

Welcome to the autonomous car project. This repository is cleanly organized to enforce reliable setups across different development laptops. It provides decoupled architectures for both background (ML testing, stream logic) and foreground (control, simulator interfaces).

## Requirements
- Python 3.10+
- Cargo & Rust (To run `rust-python` image)
- Docker
- FSDS (Formula Student Driverless Simulator) binary

---

## Architecture Overview

The system is split into two independently running Rust processes — **Foreground** and **Background** — that communicate exclusively through memory-mapped binary files in the `sharedmemory/` directory. There is **no network socket** or JSON pipe between them.

```
┌─────────────────────────────────────────────────────────────────┐
│                       FSDS SIMULATOR                            │
│                  (Formula Student Driverless)                   │
│              RPC Port 41451  ·  msgpack-rpc-python              │
└────────────┬──────────────────────────────┬────────────────────┘
             │ reads sensor data            │ receives controls
             ▼                              ▲
┌────────────────────────────┐   ┌──────────────────────────────┐
│     FOREGROUND  (Rust)     │   │     FOREGROUND  (Rust)       │
│  foreground/src/main.rs    │   │   foreground/src/main.rs     │
│                            │   │                              │
│  Spawned Python threads:   │   │  control_input_node.py       │
│  ┌──────────────────────┐  │   │  · reads control_instruction │
│  │  engine.py           │  │   │    .bin via mmap             │
│  │  · launches FSDS.exe │  │   │  · calls setCarControls()    │
│  │  · waits port 41451  │  │   │    on FSDS                   │
│  └──────────────────────┘  │   │  · watchdog: brakes if BG    │
│  ┌──────────────────────┐  │   │    packet is > 500ms old     │
│  │  imu_speed_node.py   │  │   └──────────────────────────────┘
│  │  · getCarState()     │  │
│  │  · getImuData()      │  │
│  │  · writes mmap ──────┼──┼─► sharedmemory/forground/
│  └──────────────────────┘  │       ekfin_imu_groundspeed_gyro.bin
│  ┌──────────────────────┐  │
│  │  vision_node.py      │  │
│  │  · simGetImages()    │  │
│  │  · getLidarData()    │  │
│  │  · clusters LiDAR   │  │
│  │  · writes mmap ──────┼──┼─► sharedmemory/forground/cam.bin
│  └──────────────────────┘  │   sharedmemory/forground/lid.bin
│  ┌──────────────────────┐  │
│  │ actuator_state_node  │  │
│  │  · getCarControls()  │  │
│  │  · writes mmap ──────┼──┼─► sharedmemory/forground/abs_current.bin
│  └──────────────────────┘  │
└────────────────────────────┘
```

---

## Data Flow: Node to Node

### 1 · IMU, Gyro & Ground Speed

```
FSDS Simulator
  │
  │  getImuData()          → angular_velocity (x,y,z)
  │                           linear_acceleration (x,y,z)
  │                           orientation quaternion (x,y,z,w)
  │  getCarState()         → linear_velocity magnitude → ground_speed_mps
  │
  ▼
foreground/python/imu_speed_node.py
  │
  │  Packs 52-byte struct:  format "<Q11f"
  │    [0]   timestamp_ms          uint64  (8 bytes)
  │    [1]   ground_speed_mps      float32 (4 bytes)
  │    [2-4] angular_vel x,y,z     float32 (12 bytes)
  │    [5-7] linear_acc  x,y,z     float32 (12 bytes)
  │    [8-11] orientation x,y,z,w  float32 (16 bytes)
  │
  │  mmap.write ──────────────────────────────────────────────────►
  │                sharedmemory/forground/ekfin_imu_groundspeed_gyro.bin
  ▼
background/python/test.py  (_shm_reader_thread)
  │
  │  mmap.read  ◄──────────────────────────────────────────────────
  │                sharedmemory/forground/ekfin_imu_groundspeed_gyro.bin
  │
  │  Unpacks struct → self.latest_imu dict
  │
  ▼
test.py  (refresh_ui @ 20 Hz)
  │
  │  EKF Predict step:
  │    ekf.predict(speed, yaw_rate, dt)
  │
  │  Heading correction:
  │    abs_theta = 2 * atan2(qz, qw)
  │    ekf.update_heading(abs_theta)
  │
  ▼
EKF SLAM state vector updated  ✓
```

---

### 2 · Camera Frames → YOLO → EKF Update

```
FSDS Simulator
  │
  │  simGetImages(FrontCam, Scene, pixels_as_float=False)
  │  → raw uint8 BGR image  960×540×3
  │
  ▼
foreground/python/vision_node.py
foreground/python/cam_functions/camera.py  (get_camera_frame)
  │
  │  Reshapes to (H, W, 3) numpy array
  │
foreground/python/cam_functions/shared_mem.py  (save_to_shared_memory)
  │
  │  Writes mmap binary:
  │    Header (32 bytes): QQQQ  →  height, width, channels, dtype_code=1
  │    Body:              raw uint8 bytes  (height × width × channels)
  │
  │  mmap.write ──────────────────────────────────────────────────►
  │                sharedmemory/forground/cam.bin
  ▼
background/python/cone_detection/camera_cone_detection.py
  │
  │  mmap.read  ◄──────────────────────────────────────────────────
  │                sharedmemory/forground/cam.bin
  │
  │  Unpacks QQQQ header → height, width, channels
  │  Reconstructs numpy image array
  │
  │  YOLO inference (best.pt)
  │    → bounding boxes: x1, y1, x2, y2, conf, class_label
  │    → filters: conf < 0.60, edge-touching boxes rejected
  │    → NMS via IoU > 0.3 deduplication
  │
  │  Writes mmap binary:
  │    Header (8 bytes):   Q    → num_cones
  │    Per cone (24 bytes): fffffi → x1, y1, x2, y2, conf, label_id
  │      label_id:  0=yellow  1=blue  2=orange
  │
  │  mmap.write ──────────────────────────────────────────────────►
  │                sharedmemory/background/camera_cones.bin
  ▼
background/python/test.py  (_shm_reader_thread)
  │
  │  mmap.read  ◄──────────────────────────────────────────────────
  │                sharedmemory/background/camera_cones.bin
  │
  │  Reconstructs list: [(bcx, x1, y1, x2, y2, label, conf), ...]
  │
  ▼
test.py  (_fusion_thread)
  │
  │  Camera–LiDAR fusion:
  │    · Computes camera bearing:  phi_cam = atan2(bcx - 480, 480)
  │    · Matches to LiDAR cone by angular proximity < 0.20 rad
  │    · Fused cone: { label, range (metres), bearing (rad) }
  │    · Fallback if no LiDAR match: range = (focal × real_h) / box_h
  │
  ▼
test.py  (refresh_ui @ 20 Hz)
  │
  │  EKF Update step:
  │    ekf.update(fused_measurements)
  │    → updates landmark positions in state vector
  │
  ▼
EKF SLAM map updated  ✓
```

---

### 3 · LiDAR Point Cloud → Cone Centroids

```
FSDS Simulator
  │
  │  getLidarData("Lidar", "FSCar")
  │  → raw point cloud  (x, y, z) float32 array
  │
  ▼
foreground/python/vision_node.py
foreground/python/lidar_functions/lidar.py  (detect_cones_lidar)
  │
  │  ROI filter:
  │    x ∈ [0, 20 m]       (forward only)
  │    |y| < 10 m          (lateral)
  │    z ∈ [-1.5, 1.0 m]   (height)
  │
  │  DBSCAN-style clustering  (cluster_dist = 0.30 m)
  │    → rejects clusters outside [1, 60] points
  │    → rejects clusters with spread > 1.2 m
  │    → centroid (cx, cy) per valid cluster = cone position
  │
foreground/python/lidar_functions/shared_mem.py  (save_to_shared_memory)
  │
  │  Writes mmap binary:
  │    Header (8 bytes):        Q   → num_points
  │    Per centroid (16 bytes): dd  → x (float64), y (float64)
  │
  │  mmap.write ──────────────────────────────────────────────────►
  │                sharedmemory/forground/lid.bin
  ▼
background/python/test.py  (_shm_reader_thread)
  │
  │  mmap.read  ◄──────────────────────────────────────────────────
  │                sharedmemory/forground/lid.bin
  │
  │  Unpacks Q header → num_points
  │  Reads pairs of float64 (x, y) → Euclidean distance, pixel coords
  │  → self.latest_lidar_cones list
  │
  ▼
Fusion thread  (matched against camera cones)  ✓
```

---

### 4 · Control Signals: Background → Foreground → FSDS

```
background/python/test.py  (UI sliders / keyboard / path planner)
  │
  │  self.desired = { throttle: float, brake: float, steering: float }
  │
  ▼
test.py  (_shm_writer_thread)
  │
  │  Packs 20-byte struct:  format "<Qfff"
  │    [0]  timestamp_ms    uint64  (8 bytes)  ← used by watchdog
  │    [1]  throttle        float32 (4 bytes)   range [0.0, 1.0]
  │    [2]  brake           float32 (4 bytes)   range [0.0, 1.0]
  │    [3]  steering        float32 (4 bytes)   range [-1.0, 1.0]
  │
  │  mmap.write ──────────────────────────────────────────────────►
  │                sharedmemory/background/control_instruction.bin
  ▼
foreground/python/control_input_node.py
  │
  │  mmap.read @ 20 Hz  ◄──────────────────────────────────────────
  │                sharedmemory/background/control_instruction.bin
  │
  │  Watchdog check:
  │    if now - shm_time > 500 ms  →  EMERGENCY BRAKE  (throttle=0, brake=1)
  │
  │  Clamps values:
  │    throttle  → clamp(0.0, 1.0)
  │    brake     → clamp(0.0, 1.0)
  │    steering  → clamp(-1.0, 1.0)
  │
  │  client.setCarControls(CarControls, "FSCar")
  │
  ▼
FSDS Simulator  (vehicle actuated)  ✓
```

---

## Shared Memory File Reference

All files live under `sharedmemory/` which is accessible by **both** the foreground and background processes.

| File | Direction | Format | Size |
|------|-----------|--------|------|
| `forground/cam.bin` | FG → BG | `QQQQ` header + raw uint8 BGR | 32 + H×W×3 bytes |
| `forground/lid.bin` | FG → BG | `Q` header + N×`dd` pairs | 8 + N×16 bytes |
| `forground/ekfin_imu_groundspeed_gyro.bin` | FG → BG | `<Q11f` | 52 bytes |
| `forground/abs_current.bin` | FG → BG | `<Qfff` | 20 bytes |
| `background/camera_cones.bin` | BG internal | `Q` header + N×`fffffi` | 8 + N×24 bytes |
| `background/control_instruction.bin` | BG → FG | `<Qfff` | 20 bytes |
| `background/ekf_status_matrix.bin` | BG internal | binary (reserved) | — |
| `background/lid_cam_fusion_local_cone_map.bin` | BG internal | binary (reserved) | — |

> **Note:** The `forground` directory name is intentionally spelled as-is to match the codebase.

---

## 1. Initial Setup
Clone the repository and set up your initial environments. Ensure you place the `FSDS.exe` simulator binary correctly.

1. Download `FSDS.exe` into `foreground/engine_binaries`.
2. Ensure you have moved `setting.json` to the same folder: `foreground/engine_binaries`.
3. Open your terminal in the repository root `microprocessor-code`.

## 2. Install Dependencies

### Foreground Dependencies
```powershell
cd foreground
pip install -r requirements.txt
```

### Background Dependencies
```powershell
cd background
pip install -r requirements.txt
```

## 3. Patch msgpack-rpc-python
We use an older `msgpack-rpc-python` library that crashes due to changes in Python 3+. We have provided an automated patch script to resolve this on new machines natively without needing manual source code patching!

Run the patch script from the root of the repository:
```powershell
python patch_msgpackrpc.py
```
*You should see a success message indicating `msgpackrpc` was successfully patched.*

## 4. Run the Container
We provide a Docker image to run your environment consistently. You can build and run using:

```powershell
docker build -t rust-python .

# Windows PowerShell:
docker run --rm -p 8080:80 -p 8081:81 -p 8082:82 -p 8083:83 -p 8084:84 -it -v "${PWD}:/work" -w /work rust-python bash

# macOS/Linux:
docker run --rm -p 8080:80 -p 8081:81 -p 8082:82 -p 8083:83 -p 8084:84 -it -v "$(pwd):/work" -w /work rust-python bash
```

*(If you are running the project natively without Docker, ensure your Python and Rust environments are correctly set up and skip directly to **Step 5**).*

## 5. Running the Application

Open **two separate terminal windows** in the project root.

### Terminal 1 — Foreground (Simulator + Sensor Acquisition)
```powershell
cd foreground
cargo run
```
This launches `engine.py` (starts FSDS), waits 30 s, then spawns `imu_speed_node.py`, `vision_node.py`, `actuator_state_node.py`, and `control_input_node.py` as parallel threads.

### Terminal 2 — Background (Perception + SLAM + Control Output)
```powershell
cd background
cargo run
```
This spawns `camera_cone_detection.py` (YOLO) and `test.py` (EKF SLAM dashboard) as parallel threads.

## 6. Development Workflow
Create your feature branches as `(your name)_(the function your solving)`. Make sure you are in your branch.

If you are modifying the Machine Learning implementations in `background/`, ensure that any new dependencies are manually added to `background/requirements.txt` (Do not run `pip freeze > requirements.txt` directly as it pollutes the file with local environment paths!).

Please open a Pull Request for all changes to merge into the main branch.
