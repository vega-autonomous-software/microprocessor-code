import sys
import os
import math
import time
import cv2
import pygame
import numpy as np
import engine
# Add parent folder so fsds can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import fsds
import engine
# =========================
# CONFIG
# =========================
VEHICLE_NAME = "FSCar"
CAMERA_NAME = "FrontCam"
LIDAR_NAME = "Lidar"
IMU_NAME = "Imu"

DASHBOARD_WINDOW = "FSDS Dashboard"
CONTROL_WINDOW = "Controls"

CAM_W = 960
CAM_H = 540

LIDAR_PANEL_W = 700
LIDAR_PANEL_H = 540
INFO_PANEL_W = 1660
INFO_PANEL_H = 220

DASH_W = 1660
DASH_H = 760

MAX_THROTTLE = 0.5
MAX_BRAKE = 1.0
MAX_STEERING = 0.7
STEER_STEP = 0.05
STEER_RETURN_STEP = 0.06

LIDAR_RANGE_METERS = 20.0
LIDAR_CLUSTER_DIST = 0.35
LIDAR_MIN_CLUSTER_POINTS = 3
LIDAR_MAX_CLUSTER_POINTS = 60

# Camera cone detection thresholds
MIN_CONTOUR_AREA = 120

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 120, 0)
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
GRAY = (60, 60, 60)
DARK = (20, 20, 20)

# =========================
# FSDS
# =========================
client = fsds.FSDSClient()
client.confirmConnection()
client.enableApiControl(True, VEHICLE_NAME)

car_controls = fsds.CarControls()
car_controls.throttle = 0.0
car_controls.brake = 0.0
car_controls.steering = 0.0

# =========================
# PYGAME FOR KEY INPUT
# =========================
pygame.init()
pygame.display.set_caption(CONTROL_WINDOW)
control_screen = pygame.display.set_mode((460, 140))
control_font = pygame.font.SysFont("Arial", 22)
clock = pygame.time.Clock()

# =========================
# HELPERS
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def draw_control_window(speed):
    control_screen.fill((25, 25, 25))
    lines = [
        "Focus this small window for Arrow Key control",
        "UP = throttle | DOWN = brake | LEFT/RIGHT = steer",
        "ESC = quit",
        f"Speed: {speed:.2f} m/s | Steering: {car_controls.steering:+.2f} | Throttle: {car_controls.throttle:.2f} | Brake: {car_controls.brake:.2f}",
    ]
    y = 15
    for line in lines:
        surf = control_font.render(line, True, (235, 235, 235))
        control_screen.blit(surf, (12, y))
        y += 28
    pygame.display.flip()


def update_controls():
    keys = pygame.key.get_pressed()

    # throttle / brake
    if keys[pygame.K_UP]:
        car_controls.throttle = MAX_THROTTLE
        car_controls.brake = 0.0
    elif keys[pygame.K_DOWN]:
        car_controls.throttle = 0.0
        car_controls.brake = MAX_BRAKE
    else:
        car_controls.throttle = 0.0
        car_controls.brake = 0.0

    # steering
    if keys[pygame.K_LEFT]:
        car_controls.steering -= STEER_STEP
    elif keys[pygame.K_RIGHT]:
        car_controls.steering += STEER_STEP
    else:
        if car_controls.steering > 0:
            car_controls.steering = max(0.0, car_controls.steering - STEER_RETURN_STEP)
        elif car_controls.steering < 0:
            car_controls.steering = min(0.0, car_controls.steering + STEER_RETURN_STEP)

    car_controls.steering = clamp(car_controls.steering, -MAX_STEERING, MAX_STEERING)


def get_car_speed():
    try:
        state = client.getCarState(VEHICLE_NAME)
        return state.speed
    except Exception:
        return 0.0


def get_camera_frame():
    try:
        responses = client.simGetImages(
            [
                fsds.ImageRequest(
                    camera_name=CAMERA_NAME,
                    image_type=fsds.ImageType.Scene,
                    pixels_as_float=False,
                    compress=False
                )
            ],
            vehicle_name=VEHICLE_NAME
        )
        if not responses:
            return None

        img = responses[0]
        if img.width == 0 or img.height == 0:
            return None

        arr = np.frombuffer(img.image_data_uint8, dtype=np.uint8)
        frame = arr.reshape(img.height, img.width, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    except Exception:
        return None


# =========================
# CAMERA CONE DETECTION
# =========================
def detect_cones_camera(frame):
    """
    Basic color-based cone detection.
    Returns:
      annotated_frame, detections
    detections: list of dicts with keys: x, y, w, h, color
    """
    annotated = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Approximate masks. You may need tuning depending on simulator visuals.
    masks = {
        "yellow": cv2.inRange(hsv, np.array([18, 100, 80]), np.array([40, 255, 255])),
        "blue":   cv2.inRange(hsv, np.array([90, 100, 60]), np.array([130, 255, 255])),
        "orange": cv2.inRange(hsv, np.array([5, 120, 80]), np.array([18, 255, 255])),
    }

    detections = []

    for color_name, mask in masks.items():
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Simple shape filtering: cone-like tends to be taller than wide or medium ratio
            aspect = h / max(w, 1)
            if aspect < 0.8:
                continue

            # Ignore tiny far-away noise near top
            if h < 12:
                continue

            if color_name == "yellow":
                box_color = YELLOW
            elif color_name == "blue":
                box_color = BLUE
            else:
                box_color = ORANGE

            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                annotated,
                f"{color_name} cone",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

            detections.append({
                "x": x, "y": y, "w": w, "h": h, "color": color_name,
                "cx": x + w // 2, "cy": y + h // 2
            })

    cv2.putText(
        annotated,
        f"Camera cones: {len(detections)}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        GREEN,
        2
    )

    return annotated, detections


# =========================
# LIDAR CONE DETECTION
# =========================
def euclidean_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def cluster_lidar_points(points_xy, cluster_dist=LIDAR_CLUSTER_DIST):
    """
    Simple greedy clustering.
    """
    clusters = []
    used = np.zeros(len(points_xy), dtype=bool)

    for i in range(len(points_xy)):
        if used[i]:
            continue

        cluster = [points_xy[i]]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(points_xy)):
                if used[j]:
                    continue
                for cp in cluster:
                    if euclidean_2d(points_xy[j], cp) < cluster_dist:
                        cluster.append(points_xy[j])
                        used[j] = True
                        changed = True
                        break

        clusters.append(cluster)

    return clusters


def detect_cones_lidar():
    """
    Returns:
      lidar_panel, cones
    cones: list of dicts {x, y, dist, points}
      x = forward, y = lateral
    """
    panel = np.zeros((LIDAR_PANEL_H, LIDAR_PANEL_W, 3), dtype=np.uint8)
    center_x = LIDAR_PANEL_W // 2
    center_y = LIDAR_PANEL_H - 40

    # Grid
    scale = (LIDAR_PANEL_H - 80) / LIDAR_RANGE_METERS
    for d in range(0, int(LIDAR_RANGE_METERS) + 1, 5):
        py = int(center_y - d * scale)
        cv2.line(panel, (0, py), (LIDAR_PANEL_W, py), GRAY, 1)
        cv2.putText(panel, f"{d}m", (10, max(15, py - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

    for lateral in range(-20, 21, 5):
        px = int(center_x + lateral * scale)
        cv2.line(panel, (px, 0), (px, LIDAR_PANEL_H), GRAY, 1)

    cv2.circle(panel, (center_x, center_y), 6, CYAN, -1)
    cv2.putText(panel, "Car", (center_x + 10, center_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)

    try:
        lidar = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)
        if len(lidar.point_cloud) < 3:
            cv2.putText(panel, "No lidar points", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
            return panel, []

        pts = np.array(lidar.point_cloud, dtype=np.float32).reshape(-1, 3)

        # Keep points in a forward ROI
        roi = []
        for p in pts:
            x, y, z = float(p[0]), float(p[1]), float(p[2])

            if x < 0.0 or x > LIDAR_RANGE_METERS:
                continue
            if abs(y) > 10.0:
                continue
            if z < -1.5 or z > 1.0:
                continue

            roi.append((x, y))

            px = int(center_x + y * scale)
            py = int(center_y - x * scale)
            if 0 <= px < LIDAR_PANEL_W and 0 <= py < LIDAR_PANEL_H:
                cv2.circle(panel, (px, py), 1, GREEN, -1)

        if not roi:
            return panel, []

        clusters = cluster_lidar_points(roi, LIDAR_CLUSTER_DIST)

        cone_candidates = []
        for cluster in clusters:
            n = len(cluster)
            if n < LIDAR_MIN_CLUSTER_POINTS or n > LIDAR_MAX_CLUSTER_POINTS:
                continue

            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            cx = sum(xs) / n
            cy = sum(ys) / n

            spread_x = max(xs) - min(xs)
            spread_y = max(ys) - min(ys)

            # Simple cone-sized cluster gate
            if spread_x > 0.8 or spread_y > 0.8:
                continue

            cone_candidates.append({
                "x": cx,
                "y": cy,
                "dist": math.sqrt(cx * cx + cy * cy),
                "points": n
            })

        for c in cone_candidates:
            px = int(center_x + c["y"] * scale)
            py = int(center_y - c["x"] * scale)
            cv2.circle(panel, (px, py), 7, ORANGE, 2)
            cv2.putText(panel, f"{c['dist']:.1f}m", (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1)

        cv2.putText(panel, f"Lidar cones: {len(cone_candidates)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE, 2)
        cv2.putText(panel, "Top view: forward is up", (20, LIDAR_PANEL_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        return panel, cone_candidates

    except Exception as e:
        cv2.putText(panel, f"Lidar error: {e}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        return panel, []


# =========================
# IMU PANEL
# =========================
def get_imu_info():
    try:
        imu = client.getImuData(imu_name=IMU_NAME, vehicle_name=VEHICLE_NAME)
        return imu
    except Exception:
        return None


def build_info_panel(speed, imu, cam_dets, lidar_dets):
    panel = np.zeros((INFO_PANEL_H, INFO_PANEL_W, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)

    cv2.putText(panel, "Vehicle Status", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2)

    cv2.putText(panel, f"Speed: {speed:.2f} m/s", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    cv2.putText(panel, f"Steering: {car_controls.steering:+.2f}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    cv2.putText(panel, f"Throttle: {car_controls.throttle:.2f}", (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    cv2.putText(panel, f"Brake: {car_controls.brake:.2f}", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)

    cv2.putText(panel, "Cone Detection", (420, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2)
    cv2.putText(panel, f"Camera cones: {len(cam_dets)}", (420, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, YELLOW, 2)
    cv2.putText(panel, f"Lidar cones: {len(lidar_dets)}", (420, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, ORANGE, 2)

    if cam_dets:
        preview = ", ".join([d["color"] for d in cam_dets[:6]])
        cv2.putText(panel, f"Camera colors: {preview}", (420, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)

    cv2.putText(panel, "IMU", (980, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2)

    if imu is not None:
        av = imu.angular_velocity
        la = imu.linear_acceleration
        ori = imu.orientation

        imu_lines = [
            f"Angular vel: x={av.x_val:+.3f}  y={av.y_val:+.3f}  z={av.z_val:+.3f}",
            f"Linear acc : x={la.x_val:+.3f}  y={la.y_val:+.3f}  z={la.z_val:+.3f}",
            f"Orientation: x={ori.x_val:+.3f}  y={ori.y_val:+.3f}  z={ori.z_val:+.3f}  w={ori.w_val:+.3f}",
        ]
        y = 80
        for line in imu_lines:
            cv2.putText(panel, line, (980, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2)
            y += 38
    else:
        cv2.putText(panel, "IMU unavailable", (980, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

    return panel


# =========================
# DASHBOARD ASSEMBLY
# =========================
def blank_camera_panel():
    img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cv2.putText(img, "No camera feed", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
    return img


def build_dashboard(cam_panel, lidar_panel, info_panel):
    dashboard = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)

    dashboard[0:CAM_H, 0:CAM_W] = cam_panel
    dashboard[0:LIDAR_PANEL_H, CAM_W:CAM_W + LIDAR_PANEL_W] = lidar_panel
    dashboard[CAM_H:CAM_H + INFO_PANEL_H, 0:INFO_PANEL_W] = info_panel

    cv2.putText(dashboard, "Camera", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2)
    cv2.putText(dashboard, "Lidar", (CAM_W + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2)

    return dashboard


# =========================
# MAIN
# =========================
cv2.namedWindow(DASHBOARD_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(DASHBOARD_WINDOW, DASH_W, DASH_H)

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        update_controls()
        client.setCarControls(car_controls, VEHICLE_NAME)

        speed = get_car_speed()
        draw_control_window(speed)

        frame = get_camera_frame()
        if frame is None:
            cam_panel = blank_camera_panel()
            cam_dets = []
        else:
            cam_panel, cam_dets = detect_cones_camera(frame)
            cam_panel = cv2.resize(cam_panel, (CAM_W, CAM_H))

        lidar_panel, lidar_dets = detect_cones_lidar()
        imu = get_imu_info()
        info_panel = build_info_panel(speed, imu, cam_dets, lidar_dets)

        dashboard = build_dashboard(cam_panel, lidar_panel, info_panel)
        cv2.imshow(DASHBOARD_WINDOW, dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt

        clock.tick(30)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    try:
        car_controls.throttle = 0.0
        car_controls.brake = 1.0
        car_controls.steering = 0.0
        client.setCarControls(car_controls, VEHICLE_NAME)
        client.enableApiControl(False, VEHICLE_NAME)
    except Exception:
        pass

    cv2.destroyAllWindows()
    pygame.quit()