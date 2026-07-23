import numpy as np
import math


# This EKF uses a dense covariance matrix: update cost grows quadratically
# with landmark count.  Keep a small local working set here; the dashboard's
# candidate map stores the complete persistent cone history.
MAX_ACTIVE_EKF_LANDMARKS = 120


class EKFSLAM:
    def __init__(self, R_pose=None, Q_obs=None, init_p_pose=1e-3, init_p_landmark=5.0, assoc_threshold=3.0):
        """
        EKF SLAM core state estimation.
        - State vector x: [x_v, y_v, theta_v, x_1, y_1, ..., x_M, y_M]^T
        - Covariance matrix P
        """
        # Vehicle pose initialized at origin: [0, 0, 0]
        self.x = np.zeros(3, dtype=np.float64)

        # Covariance initialized to small values for vehicle pose
        self.P = np.diag([init_p_pose, init_p_pose, init_p_pose]).astype(np.float64)

        # Process noise covariance (for vehicle pose prediction step)
        # R_pose: noise in [x, y, theta] propagation
        if R_pose is None:
            # Drastically lowered process noise (Kinematic Trust increased)
            self.R_pose = np.diag([0.005**2, 0.005**2, math.radians(0.5)**2]).astype(np.float64)
        else:
            self.R_pose = np.array(R_pose, dtype=np.float64)

        # Measurement noise covariance
        # Q_obs: noise in [range, bearing] observation
        if Q_obs is None:
            # Increased noise to absorb monocular depth fluctuations and prevent landmark scattering
            self.Q_obs = np.diag([1.0**2, math.radians(5.0)**2]).astype(np.float64)
        else:
            self.Q_obs = np.array(Q_obs, dtype=np.float64)

        self.init_p_landmark = init_p_landmark
        self.assoc_threshold = assoc_threshold

        # Landmark tracker to store association info and colors
        # list of dict: {"id": i, "color": color}
        self.landmarks = []
        
        # Provisional landmarks: {"x": float, "y": float, "color": str, "hit_count": int}
        self.provisional_landmarks = []
        self.heading_initialized = False

        # Keep track of history trace for trajectory rendering
        self.trajectory = []
        self.trajectory.append((float(self.x[0]), float(self.x[1])))

    def can_track_candidate(self, candidate_id):
        """Whether this candidate belongs to the local dense-EKF working set."""
        return (
            any(item.get("candidate_id") == candidate_id for item in self.landmarks)
            or len(self.landmarks) < MAX_ACTIVE_EKF_LANDMARKS
        )

    def retain_candidate_ids(self, candidate_ids):
        """Drop distant landmarks from the dense EKF, preserving global map data.

        The caller owns the lightweight global candidate map.  Removing an
        inactive landmark here only reduces the local covariance matrix; it
        never removes the corresponding cone from that persistent map.
        """
        keep = [
            index for index, item in enumerate(self.landmarks)
            if item.get("candidate_id") in candidate_ids
        ]
        if len(keep) == len(self.landmarks):
            return

        state_indices = [0, 1, 2]
        for index in keep:
            state_indices.extend((3 + 2 * index, 4 + 2 * index))
        self.x = self.x[state_indices]
        self.P = self.P[np.ix_(state_indices, state_indices)]
        self.landmarks = [dict(self.landmarks[index], id=new_id) for new_id, index in enumerate(keep)]

    def predict(self, v, omega, dt):
        """
        Predict vehicle pose using ground speed and yaw rate.
        v: ground speed (m/s)
        omega: angular velocity (rad/s)
        dt: time step (seconds)
        """
        if dt <= 0:
            return

        # Current state info
        xv, yv, theta = self.x[0], self.x[1], self.x[2]

        # 1. Update vehicle state
        self.x[0] = xv + v * math.cos(theta) * dt
        self.x[1] = yv + v * math.sin(theta) * dt
        self.x[2] = self.normalize_angle(theta + omega * dt)

        # 2. Update Covariance Matrix P
        # Jacobian G_t of motion model with respect to full state
        n_states = len(self.x)
        G = np.eye(n_states, dtype=np.float64)
        G[0, 2] = -v * math.sin(theta) * dt
        G[1, 2] = v * math.cos(theta) * dt

        # Predict covariance: P = G * P * G^T + R_t
        self.P = G @ self.P @ G.T
        
        # Scale process noise by velocity and dt to prevent stationary covariance inflation
        # If the car is stopped, process noise is almost zero.
        noise_scale = dt * max(0.1, abs(v))
        self.P[0:3, 0:3] += self.R_pose * noise_scale

        # Store historical trace if car has moved significantly
        last_pt = self.trajectory[-1]
        if math.hypot(self.x[0] - last_pt[0], self.x[1] - last_pt[1]) > 0.05:
            self.trajectory.append((float(self.x[0]), float(self.x[1])))
            if len(self.trajectory) > 600:
                self.trajectory.pop(0)

    def update_heading(self, abs_theta, heading_variance=math.radians(2.0) ** 2):
        """
        Fuse absolute IMU heading as a scalar EKF measurement.
        """
        H = np.zeros((1, len(self.x)), dtype=np.float64)
        H[0, 2] = 1.0
        innovation = self.normalize_angle(abs_theta - self.x[2])
        S = float((H @ self.P @ H.T)[0, 0] + heading_variance)
        if S <= 0.0:
            return
        K = (self.P @ H.T) / S
        self.x = self.x + K[:, 0] * innovation
        identity = np.eye(len(self.x), dtype=np.float64)
        correction = identity - K @ H
        self.P = correction @ self.P @ correction.T + (K * heading_variance) @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x[2] = self.normalize_angle(self.x[2])

    def update(self, fused_measurements):
        """
        Update state using a list of fused measurements.
        fused_measurements: list of dicts: {"range": float, "bearing": float, "color": str}
        """
        used_landmarks = set()
        for meas in fused_measurements:
            r = meas.get("range")
            b = meas.get("bearing")
            color = meas.get("color")
            if b is None or color is None:
                continue
            if r is not None and r > 15.0:
                continue

            matched_index = self._update_single(
                r,
                b,
                color,
                meas.get("candidate_id"),
                used_landmarks,
            )
            if matched_index is not None:
                used_landmarks.add(matched_index)

    def _update_single(self, r, b, color, candidate_id=None, used_landmarks=None):
        xv, yv, theta = self.x[0], self.x[1], self.x[2]

        n_landmarks = len(self.landmarks)
        known_candidate_exists = (
            candidate_id is not None
            and any(
                landmark.get("candidate_id") == candidate_id
                for landmark in self.landmarks
            )
        )

        # Data association is geometric. Colour is semantic evidence, not a hard
        # gate, because a distant classifier result may later be corrected.
        best_idx = -1
        best_dist = self.assoc_threshold
        used_landmarks = used_landmarks or set()

        for idx in range(n_landmarks):
            l_info = self.landmarks[idx]
            if idx in used_landmarks:
                continue

            lx_idx = 3 + 2 * idx
            ly_idx = 4 + 2 * idx
            lx, ly = self.x[lx_idx], self.x[ly_idx]

            # Predicted measurement for this landmark
            dx = lx - xv
            dy = ly - yv
            d2 = dx**2 + dy**2
            d = math.sqrt(d2)

            if d < 1e-5:
                continue

            pred_r = d
            pred_b = self.normalize_angle(math.atan2(dy, dx) - theta)

            # Measurement Jacobian H for this landmark
            H = np.zeros((2, len(self.x)), dtype=np.float64)
            # Derivative w.r.t vehicle pose
            H[0, 0] = -dx / d
            H[0, 1] = -dy / d
            H[0, 2] = 0.0
            H[1, 0] = dy / d2
            H[1, 1] = -dx / d2
            H[1, 2] = -1.0

            # Derivative w.r.t landmark coordinates
            H[0, lx_idx] = dx / d
            H[0, ly_idx] = dy / d
            H[1, lx_idx] = -dy / d2
            H[1, ly_idx] = dx / d2

            if r is not None:
                y_val = np.array([r - pred_r, self.normalize_angle(b - pred_b)], dtype=np.float64)
                H_i = H
                Q_i = self.Q_obs
            else:
                # Bearing only update
                y_val = np.array([self.normalize_angle(b - pred_b)], dtype=np.float64)
                H_i = H[1:2, :]
                Q_i = self.Q_obs[1:2, 1:2]

            # Innovation covariance: S = H * P * H^T + Q
            S = H_i @ self.P @ H_i.T + Q_i

            # Mahalanobis Distance
            try:
                S_inv = np.linalg.inv(S) if r is not None else np.array([[1.0 / S[0, 0]]])
                d_M = math.sqrt(y_val @ S_inv @ y_val)
            except np.linalg.LinAlgError:
                d_M = 999.0

            same_candidate = (
                candidate_id is not None
                and l_info.get("candidate_id") == candidate_id
            )
            if d_M < self.assoc_threshold and same_candidate:
                best_dist = -1.0
                best_idx = idx
            elif best_dist >= 0.0 and d_M < best_dist:
                best_dist = d_M
                best_idx = idx

        if best_idx != -1:
            # Associate and update existing landmark
            idx = best_idx
            lx_idx = 3 + 2 * idx
            ly_idx = 4 + 2 * idx
            lx, ly = self.x[lx_idx], self.x[ly_idx]

            dx = lx - xv
            dy = ly - yv
            d2 = dx**2 + dy**2
            d = math.sqrt(d2)

            pred_r = d
            pred_b = self.normalize_angle(math.atan2(dy, dx) - theta)

            # Recompute Jacobian H
            H = np.zeros((2, len(self.x)), dtype=np.float64)
            H[0, 0] = -dx / d
            H[0, 1] = -dy / d
            H[0, 2] = 0.0
            H[1, 0] = dy / d2
            H[1, 1] = -dx / d2
            H[1, 2] = -1.0

            H[0, lx_idx] = dx / d
            H[0, ly_idx] = dy / d
            H[1, lx_idx] = -dy / d2
            H[1, ly_idx] = dx / d2

            if r is not None:
                y_val = np.array([r - pred_r, self.normalize_angle(b - pred_b)], dtype=np.float64)
                H_i = H
                Q_i = self.Q_obs
            else:
                y_val = np.array([self.normalize_angle(b - pred_b)], dtype=np.float64)
                H_i = H[1:2, :]
                Q_i = self.Q_obs[1:2, 1:2]

            S = H_i @ self.P @ H_i.T + Q_i
            try:
                S_inv = np.linalg.inv(S) if r is not None else np.array([[1.0 / S[0, 0]]])
                K = self.P @ H_i.T @ S_inv
                self.x = self.x + K @ y_val
                identity = np.eye(len(self.x), dtype=np.float64)
                correction = identity - K @ H_i
                self.P = correction @ self.P @ correction.T + K @ Q_i @ K.T
                self.P = 0.5 * (self.P + self.P.T)
                self.x[2] = self.normalize_angle(self.x[2])
            except np.linalg.LinAlgError:
                pass
                
            self.landmarks[idx]["hit_count"] += 1
            self.landmarks[idx]["color"] = color
            if candidate_id is not None:
                self.landmarks[idx]["candidate_id"] = candidate_id
            return idx

        elif known_candidate_exists:
            # A tracked landmark produced an incompatible outlier. Reject it
            # rather than duplicating the same physical cone in the map.
            return None

        elif r is not None and len(self.landmarks) < MAX_ACTIVE_EKF_LANDMARKS:
            angle = theta + b
            xl_est = xv + r * math.cos(angle)
            yl_est = yv + r * math.sin(angle)

            old_size = len(self.x)
            G_state = np.zeros((2, old_size), dtype=np.float64)
            G_state[:, 0:3] = np.array([
                [1.0, 0.0, -r * math.sin(angle)],
                [0.0, 1.0, r * math.cos(angle)],
            ])
            G_measurement = np.array([
                [math.cos(angle), -r * math.sin(angle)],
                [math.sin(angle), r * math.cos(angle)],
            ])

            landmark_cross = G_state @ self.P
            landmark_covariance = (
                G_state @ self.P @ G_state.T
                + G_measurement @ self.Q_obs @ G_measurement.T
            )
            new_P = np.zeros((old_size + 2, old_size + 2), dtype=np.float64)
            new_P[:old_size, :old_size] = self.P
            new_P[old_size:, :old_size] = landmark_cross
            new_P[:old_size, old_size:] = landmark_cross.T
            new_P[old_size:, old_size:] = landmark_covariance

            self.x = np.append(self.x, [xl_est, yl_est])
            self.P = 0.5 * (new_P + new_P.T)
            self.landmarks.append({
                "id": n_landmarks,
                "candidate_id": candidate_id,
                "color": color,
                "hit_count": 1,
            })
            return n_landmarks

        return None

    @staticmethod
    def normalize_angle(angle):
        """Normalize an angle to [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
