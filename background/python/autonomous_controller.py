import math
import time
import traceback
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

PATH_START_THROTTLE = 0.30
PATH_START_DURATION_S = 1.0
PATH_MOVING_SPEED_MPS = 0.20
MAX_SAFE_STEERING_ANGLE = math.radians(30.0)
MAX_PATH_HEADING_ERROR = math.radians(65.0)
MAX_STEERING_STEP = 0.08
LAST_VALID_PATH_HOLD_S = 0.40
TRACK_GATE_MAX_FORWARD_OFFSET_M = 2.5
TRACK_GATE_MAX_WIDTH_M = 5.5
TRACK_SHAPE_MERGE_DISTANCE_M = 1.0
TRACK_SHAPE_MAX_EXTENSION_GAP_M = 7.0
TRACK_SHAPE_MIN_EXTENSION_ALIGNMENT = 0.25

class AutonomousController:
    def __init__(self, target_speed=4.0, max_path_length=15.0):
        # Kinematic Bicycle Model Constraints
        self.L = 1.53  # Wheelbase (m)
        self.max_steer_angle = math.radians(45.0)
        self.max_safe_steering_angle = MAX_SAFE_STEERING_ANGLE
        self.max_path_heading_error = MAX_PATH_HEADING_ERROR
        self.max_acceleration = 8.0  # m/s^2
        self.max_deceleration = 11.0  # m/s^2
        
        # Stanley Tuning Parameters
        self.k_e = 1.0  # Cross-track error gain (increase for tighter cornering)
        self.k_s = 1.0  # Softening constant (increase to prevent low-speed wobble)
        self.lookahead_dist = 1.5  # Heading lookahead (meters)
        # Maximum cumulative length of the planned centreline.  This also
        # bounds the purple route shown on the EKF map.
        self.max_path_length = max(0.1, float(max_path_length))
        
        self.target_speed = target_speed
        
        # State tracking
        self.last_target_point = None
        self.last_waypoints = []
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_steering = 0.0
        self.progressive_path = False
        self.path_started_at = None
        self.path_evidence_count = 0
        self.last_status = "IDLE"
        self.path_turn_rejected = False
        self._last_error = None
        self.last_valid_path_at = None
        # Persistent, ordered centreline built from conservative local cone
        # pairs.  It grows during the first lap and is refined on every later
        # observation; it is not delayed until a lap finishes.
        self.track_shape = []
        self.track_shape_hits = []
        self.track_shape_closed = False
        self._observed_midpoints = []
        self.using_learned_route = False

    def _report_error(self, stage, error):
        """Print each distinct controller failure once instead of hiding it."""
        key = (stage, type(error).__name__, str(error))
        if key != self._last_error:
            self._last_error = key
            print(f"[Autonomy] {stage} failed: {error}", flush=True)
            traceback.print_exc()

    def get_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def limit_path_length(self, waypoints):
        """Keep no more than ``max_path_length`` metres of a waypoint path.

        If the final segment crosses the limit, add an interpolated endpoint
        so the displayed and controlled path ends exactly at the configured
        distance instead of jumping past it.
        """
        if len(waypoints) < 2:
            return list(waypoints)

        limited = [waypoints[0]]
        remaining = self.max_path_length
        for point in waypoints[1:]:
            start = limited[-1]
            segment_length = self.get_distance(start, point)
            if segment_length < 1e-9:
                continue
            if segment_length <= remaining:
                limited.append(point)
                remaining -= segment_length
                continue

            fraction = remaining / segment_length
            limited.append((
                start[0] + (point[0] - start[0]) * fraction,
                start[1] + (point[1] - start[1]) * fraction,
            ))
            break
        return limited

    def extract_centerline(self, ekf_state, vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0, max_track_width=20.0):
        """
        Uses Delaunay Triangulation to extract the centerline between blue and yellow cones.
        """
        self.path_evidence_count = 0
        blue_cones = [c for c in ekf_state if 'blue' in c['color']]
        yellow_cones = [c for c in ekf_state if 'yellow' in c['color']]
        all_cones = blue_cones + yellow_cones
        
        # Local Horizon Fix: Only triangulate cones in front of the car
        cones = []
        for c in all_cones:
            dx = c['x'] - vehicle_x
            dy = c['y'] - vehicle_y
            local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
            if local_x > -1.0 and math.hypot(dx, dy) < 25.0:
                cones.append(c)
                
        waypoints = []
        
        if len(cones) >= 3:
            points = np.array([[c['x'], c['y']] for c in cones])
            colors = [c['color'] for c in cones]
            
            try:
                tri = Delaunay(points)
                cross_edges = set()
                
                for simplex in tri.simplices:
                    for i in range(3):
                        idx1 = simplex[i]
                        idx2 = simplex[(i + 1) % 3]
                        c1, c2 = colors[idx1], colors[idx2]
                        
                        if ('blue' in c1 and 'yellow' in c2) or ('yellow' in c1 and 'blue' in c2):
                            dist = self.get_distance(points[idx1], points[idx2])
                            if dist <= max_track_width:
                                cross_edges.add(tuple(sorted((idx1, idx2))))

                if not cross_edges:
                    for simplex in tri.simplices:
                        for i in range(3):
                            idx1 = simplex[i]
                            idx2 = simplex[(i + 1) % 3]
                            c1, c2 = colors[idx1], colors[idx2]
                            if ('blue' in c1 and 'yellow' in c2) or ('yellow' in c1 and 'blue' in c2):
                                dist = self.get_distance(points[idx1], points[idx2])
                                if dist <= max_track_width * 1.5:
                                    cross_edges.add(tuple(sorted((idx1, idx2))))
                                 
                for idx1, idx2 in cross_edges:
                    c1, c2 = colors[idx1], colors[idx2]
                    mx = (points[idx1][0] + points[idx2][0]) / 2.0
                    my = (points[idx1][1] + points[idx2][1]) / 2.0
            
                    if 'blue' in c1:
                        p_blue, p_yellow = points[idx1], points[idx2]
                    else:
                        p_blue, p_yellow = points[idx2], points[idx1]
                
                    v_dx = p_yellow[0] - p_blue[0]
                    v_dy = p_yellow[1] - p_blue[1]
            
                    fwd_dx = v_dy
                    fwd_dy = -v_dx
            
                    mag = math.hypot(fwd_dx, fwd_dy)
                    if mag > 0.001:
                        fwd_dx /= mag
                        fwd_dy /= mag
                        waypoints.append((mx, my, fwd_dx, fwd_dy))
            except Exception as error:
                self._report_error("Delaunay centreline", error)

        if not waypoints:
            return []

        # Directional Nearest Neighbor Sorting
        ordered_waypoints = []
        if waypoints:
            current_pt = min(waypoints, key=lambda p: math.hypot(p[0] - vehicle_x, p[1] - vehicle_y))
            ordered_waypoints.append(current_pt)
            unvisited = set(waypoints)
            unvisited.remove(current_pt)
            
            while unvisited:
                track_dx = current_pt[2]
                track_dy = current_pt[3]
                
                best_pt = None
                best_score = float('inf')
                
                for pt in unvisited:
                    vx = pt[0] - current_pt[0]
                    vy = pt[1] - current_pt[1]
                    dist = math.hypot(vx, vy)
                    
                    if dist > 0.001:
                        vx /= dist
                        vy /= dist
                    
                    dot = track_dx * vx + track_dy * vy
                    if dot < 0.0:
                        continue
                        
                    score = dist
                    if score < best_score:
                        best_score = score
                        best_pt = pt
                        
                if best_pt is None or best_score > 30.0:
                    break
                    
                ordered_waypoints.append(best_pt)
                unvisited.remove(best_pt)
                current_pt = best_pt

        self.path_evidence_count = len(ordered_waypoints)

        # B-Spline Smoothing
        if len(ordered_waypoints) >= 4:
            try:
                pts = np.array([(p[0], p[1]) for p in ordered_waypoints])
                _, idx = np.unique(pts, axis=0, return_index=True)
                pts = pts[np.sort(idx)]
                
                if len(pts) >= 4:
                    tck, u = splprep([pts[:, 0], pts[:, 1]], s=3.0, k=3)
                    u_new = np.linspace(0, 1, len(pts) * 3)
                    new_points = splev(u_new, tck)
                    ordered_waypoints = list(zip(new_points[0], new_points[1]))
                else:
                    ordered_waypoints = [(p[0], p[1]) for p in ordered_waypoints]
            except Exception as error:
                self._report_error("Centreline smoothing", error)
                ordered_waypoints = [(p[0], p[1]) for p in ordered_waypoints]
        else:
            ordered_waypoints = [(p[0], p[1]) for p in ordered_waypoints]

        ordered_waypoints = self.limit_path_length(ordered_waypoints)
        self.last_waypoints = ordered_waypoints
        return ordered_waypoints

    def extract_progressive_centerline(self, ekf_state, vehicle_x, vehicle_y, vehicle_theta):
        """Build the best short path available before a full centreline exists."""
        orange = [c for c in ekf_state if "orange" in c["color"]]
        blue = [c for c in ekf_state if "blue" in c["color"]]
        yellow = [c for c in ekf_state if "yellow" in c["color"]]
        cos_h, sin_h = math.cos(vehicle_theta), math.sin(vehicle_theta)
        anchors, used_yellow = [], set()

        # Only accept a true left/right gate in the vehicle frame.  Pairing by
        # raw Euclidean distance lets cones across a hairpin become a false
        # gate, which was the source of the misplaced purple point.
        for left in blue:
            left_dx, left_dy = left["x"] - vehicle_x, left["y"] - vehicle_y
            left_forward = left_dx * cos_h + left_dy * sin_h
            left_lateral = -left_dx * sin_h + left_dy * cos_h
            if left_lateral <= 0.0:
                continue
            options = []
            for right in yellow:
                if id(right) in used_yellow:
                    continue
                right_dx, right_dy = right["x"] - vehicle_x, right["y"] - vehicle_y
                right_forward = right_dx * cos_h + right_dy * sin_h
                right_lateral = -right_dx * sin_h + right_dy * cos_h
                if right_lateral >= 0.0:
                    continue
                width = self.get_distance((left["x"], left["y"]), (right["x"], right["y"]))
                mid = ((left["x"] + right["x"]) / 2.0, (left["y"] + right["y"]) / 2.0)
                local_x = (mid[0] - vehicle_x) * cos_h + (mid[1] - vehicle_y) * sin_h
                if (
                    # A cross-track gate must look like the actual track,
                    # not two cones from adjacent legs of a hairpin.  The
                    # latter was wide enough to pass the old 8 m gate and
                    # produced a single purple waypoint off the road.
                    1.5 <= width <= TRACK_GATE_MAX_WIDTH_M
                    and 0.5 <= local_x <= 15.0
                    and abs(left_forward - right_forward) <= TRACK_GATE_MAX_FORWARD_OFFSET_M
                ):
                    options.append((abs(left_forward - right_forward), width, local_x, mid, right))
            if options:
                _, _, local_x, midpoint, right = min(options, key=lambda item: (item[0], item[1]))
                used_yellow.add(id(right))
                anchors.append((local_x, midpoint))

        # Orange cones are an optional centre reference, never a requirement.
        visible_orange = []
        for cone in orange:
            local_x = (cone["x"] - vehicle_x) * cos_h + (cone["y"] - vehicle_y) * sin_h
            if -1.0 <= local_x <= 15.0:
                visible_orange.append((local_x, cone))
        if len(visible_orange) >= 2:
            group = sorted(visible_orange, key=lambda item: abs(item[0]))[:4]
            centre = (
                sum(item[1]["x"] for item in group) / len(group),
                sum(item[1]["y"] for item in group) / len(group),
            )
            centre_x = (centre[0] - vehicle_x) * cos_h + (centre[1] - vehicle_y) * sin_h
            anchors.append((centre_x, centre))

        if not anchors:
            self._observed_midpoints = []
            self.path_evidence_count = 0
            self.last_status = "WAITING: no usable centreline waypoint"
            return []

        # Nearby observations represent the same path point; merge rather than stop.
        merged = []
        for _, point in sorted(anchors, key=lambda item: item[0]):
            if merged and self.get_distance(merged[-1], point) < 0.5:
                merged[-1] = ((merged[-1][0] + point[0]) / 2.0, (merged[-1][1] + point[1]) / 2.0)
            else:
                merged.append(point)

        self._observed_midpoints = list(merged)

        waypoints = [(vehicle_x, vehicle_y)] + merged
        reference = waypoints[-2]
        dx, dy = waypoints[-1][0] - reference[0], waypoints[-1][1] - reference[1]
        distance = math.hypot(dx, dy)
        if distance < 0.2:
            dx, dy, distance = cos_h, sin_h, 1.0
        ux, uy = dx / distance, dy / distance
        extension = max(2.0, self.lookahead_dist)
        waypoints.append((waypoints[-1][0] + ux * extension, waypoints[-1][1] + uy * extension))
        self.path_evidence_count = len(merged)
        waypoints = self.limit_path_length(waypoints)
        self.last_waypoints = waypoints
        return waypoints

    def _update_track_shape(self, vehicle_x, vehicle_y, vehicle_theta):
        """Continuously extend/refine the ordered track shape from local gates."""
        if not self._observed_midpoints:
            return

        for point in self._observed_midpoints:
            if not self.track_shape:
                self.track_shape.append(point)
                self.track_shape_hits.append(1)
                continue

            distances = [self.get_distance(point, existing) for existing in self.track_shape]
            closest_index = int(np.argmin(distances))
            if distances[closest_index] <= TRACK_SHAPE_MERGE_DISTANCE_M:
                hits = self.track_shape_hits[closest_index]
                existing = self.track_shape[closest_index]
                self.track_shape[closest_index] = (
                    (existing[0] * hits + point[0]) / (hits + 1),
                    (existing[1] * hits + point[1]) / (hits + 1),
                )
                self.track_shape_hits[closest_index] = hits + 1
                continue

            # Once a lap has supplied a closed centreline, it is the route
            # prior for every later lap.  New observations improve existing
            # points, but must never grow a competing branch: the controller
            # should already know whether the road turns left or right before
            # those next cones come into view.
            if self.track_shape_closed:
                continue

            endpoint = self.track_shape[-1]
            extension_x, extension_y = point[0] - endpoint[0], point[1] - endpoint[1]
            extension_length = math.hypot(extension_x, extension_y)
            extension_alignment = (
                (extension_x * math.cos(vehicle_theta) + extension_y * math.sin(vehicle_theta))
                / extension_length
                if extension_length > 1e-6 else 1.0
            )
            # A map point can only grow the route in the direction the car is
            # currently travelling.  This keeps one bad pair from becoming a
            # permanent branch in the learned track shape.
            if (
                extension_length <= TRACK_SHAPE_MAX_EXTENSION_GAP_M
                and extension_alignment >= TRACK_SHAPE_MIN_EXTENSION_ALIGNMENT
            ):
                self.track_shape.append(point)
                self.track_shape_hits.append(1)

        # Closure is only needed for wrap-around indexing.  The growing shape
        # is already used for planning long before this becomes true.
        if len(self.track_shape) >= 12:
            shape_length = sum(
                self.get_distance(first, second)
                for first, second in zip(self.track_shape, self.track_shape[1:])
            )
            if shape_length >= 30.0 and self.get_distance(self.track_shape[-1], self.track_shape[0]) <= 4.0:
                self.track_shape_closed = True

    def _path_from_track_shape(self, vehicle_x, vehicle_y, vehicle_theta):
        """Return the forward portion of the dynamically learned centreline."""
        count = len(self.track_shape)
        if count < 2:
            return []

        heading_x, heading_y = math.cos(vehicle_theta), math.sin(vehicle_theta)
        candidates = []
        maximum = count if self.track_shape_closed else count - 1
        for index in range(maximum):
            point = self.track_shape[index]
            following = self.track_shape[(index + 1) % count]
            dx, dy = point[0] - vehicle_x, point[1] - vehicle_y
            forward = dx * heading_x + dy * heading_y
            tangent_x, tangent_y = following[0] - point[0], following[1] - point[1]
            tangent_length = math.hypot(tangent_x, tangent_y)
            if forward < -1.0 or tangent_length < 0.1:
                continue
            alignment = (tangent_x * heading_x + tangent_y * heading_y) / tangent_length
            if alignment >= 0.20:
                candidates.append((math.hypot(dx, dy), -alignment, index))
        if not candidates:
            return []

        _, _, start = min(candidates)
        path = [(vehicle_x, vehicle_y)]
        for offset in range(count):
            index = start + offset
            if self.track_shape_closed:
                index %= count
            elif index >= count:
                break
            point = self.track_shape[index]
            if self.get_distance(path[-1], point) >= 0.25:
                path.append(point)
            if len(path) > 1 and sum(
                self.get_distance(first, second) for first, second in zip(path, path[1:])
            ) >= self.max_path_length:
                break

        if len(path) < 3:
            return []
        self.path_evidence_count = len(path) - 1
        return self.limit_path_length(path)

    def _path_is_forward_compatible(self, waypoints, vehicle_x, vehicle_y, vehicle_theta):
        """Whether the route can be entered without an unsafe steering reversal.

        The persistent shape is intentionally retained across laps, but near
        the start/finish join its nearest segment can temporarily be the
        segment just driven instead of the segment being entered.  In that
        case use the current LiDAR cone gates rather than braking on a route
        which is geometrically valid but faces the wrong way.
        """
        if len(waypoints) < 2:
            return False

        target_index = next((
            index for index, point in enumerate(waypoints)
            if (point[0] - vehicle_x) * math.cos(vehicle_theta)
            + (point[1] - vehicle_y) * math.sin(vehicle_theta) > 0.05
        ), None)
        if target_index is None:
            return False
        target = waypoints[target_index]
        following = next((
            point for point in waypoints[target_index + 1:]
            if self.get_distance(point, target) >= 0.25
        ), None)
        if following is None:
            path_yaw = math.atan2(target[1] - vehicle_y, target[0] - vehicle_x)
        else:
            path_yaw = math.atan2(following[1] - target[1], following[0] - target[0])
        heading_error = math.atan2(
            math.sin(path_yaw - vehicle_theta),
            math.cos(path_yaw - vehicle_theta),
        )
        return abs(heading_error) <= self.max_path_heading_error

    def mpc_control(self, current_speed, target_speed):
        """Optimize a short acceleration sequence for longitudinal speed."""
        horizon, dt = 5, 0.2
        initial = np.zeros(horizon)
        bounds = [(-self.max_deceleration, self.max_acceleration)] * horizon

        def cost_fn(accelerations):
            cost, speed = 0.0, current_speed
            for index, acceleration in enumerate(accelerations):
                speed += acceleration * dt
                cost += 10.0 * (speed - target_speed) ** 2 + acceleration ** 2
                if index > 0:
                    cost += 5.0 * (acceleration - accelerations[index - 1]) ** 2
            return cost

        try:
            result = minimize(cost_fn, initial, bounds=bounds, method="SLSQP")
            acceleration = result.x[0] if result.success else 1.5 * (target_speed - current_speed)
        except Exception as error:
            self._report_error("Longitudinal controller", error)
            acceleration = 1.5 * (target_speed - current_speed)

        acceleration = max(-self.max_deceleration, min(self.max_acceleration, acceleration))
        if acceleration > 0.0:
            return acceleration / self.max_acceleration, 0.0
        return 0.0, -acceleration / self.max_deceleration

    def stanley_control(self, vehicle_x, vehicle_y, vehicle_theta, vehicle_speed, waypoints):
        """Calculate lateral steering from heading and cross-track error."""
        self.path_turn_rejected = False
        if not waypoints:
            self.last_target_point = None
            return 0.0
        closest_idx = target_local_y = None
        for index, point in enumerate(waypoints):
            dx, dy = point[0] - vehicle_x, point[1] - vehicle_y
            local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
            local_y = -dx * math.sin(vehicle_theta) + dy * math.cos(vehicle_theta)
            if local_x > 0.0:
                closest_idx, target_local_y = index, local_y
                break
        if closest_idx is None:
            self.last_target_point = None
            return 0.0

        target = waypoints[closest_idx]
        self.last_target_point = target
        lookahead_idx = closest_idx
        for index in range(closest_idx + 1, len(waypoints)):
            if self.get_distance(waypoints[index], target) >= self.lookahead_dist:
                lookahead_idx = index
                break
        if lookahead_idx > closest_idx:
            following = waypoints[lookahead_idx]
            path_yaw = math.atan2(following[1] - target[1], following[0] - target[0])
        else:
            path_yaw = vehicle_theta

        heading_error = (path_yaw - vehicle_theta + math.pi) % (2 * math.pi) - math.pi
        # A path that points far away from the vehicle heading is usually a
        # bad association or a path loop. Stop instead of initiating a U-turn.
        if abs(heading_error) > self.max_path_heading_error:
            self.path_turn_rejected = True
            return 0.0
        safe_speed = max(1.0, vehicle_speed)
        steering = -(heading_error + math.atan2(self.k_e * target_local_y, safe_speed + self.k_s))
        safe_limit = min(self.max_steer_angle, self.max_safe_steering_angle)
        return max(-1.0, min(1.0, steering / safe_limit))

    def _slew_limit_steering(self, requested):
        """Prevent one noisy path update from causing a steering snap."""
        delta = max(-MAX_STEERING_STEP, min(MAX_STEERING_STEP, requested - self.last_steering))
        return self.last_steering + delta

    def compute_commands(self, ekf_state, vehicle_x, vehicle_y, vehicle_theta, vehicle_speed):
        # Learn from strict local gates every cycle.  The model immediately
        # supplies the reliable already-seen part of the track; fresh local
        # gates extend it as the car reaches new territory.
        progressive_waypoints = self.extract_progressive_centerline(
            ekf_state, vehicle_x, vehicle_y, vehicle_theta,
        )
        self._update_track_shape(vehicle_x, vehicle_y, vehicle_theta)
        track_waypoints = self._path_from_track_shape(
            vehicle_x, vehicle_y, vehicle_theta,
        )
        track_is_compatible = self._path_is_forward_compatible(
            track_waypoints, vehicle_x, vehicle_y, vehicle_theta,
        )
        progressive_is_compatible = self._path_is_forward_compatible(
            progressive_waypoints, vehicle_x, vehicle_y, vehicle_theta,
        )
        # Fresh local gates win when the retained route is momentarily facing
        # backwards at the lap join.  This avoids the safety-stop seen just
        # before the first lap completes while retaining the global shape for
        # the normal case.
        if track_is_compatible:
            waypoints = track_waypoints
            self.progressive_path = False
            self.using_learned_route = self.track_shape_closed
        elif progressive_is_compatible:
            waypoints = progressive_waypoints
            self.progressive_path = True
            self.using_learned_route = False
        else:
            waypoints = []
            self.progressive_path = True
            self.using_learned_route = False
        if not waypoints:
            # Legacy Delaunay is a final fallback only; it is no longer the
            # main source of purple waypoints.
            waypoints = self.extract_centerline(
                ekf_state, vehicle_x, vehicle_y, vehicle_theta,
            )
        now = time.monotonic()
        if waypoints:
            self.last_waypoints = list(waypoints)
            self.last_valid_path_at = now
        elif (
            self.last_valid_path_at is not None
            and now - self.last_valid_path_at <= LAST_VALID_PATH_HOLD_S
            and self.last_waypoints
        ):
            # A single dropped scan should not turn into a brake/steer spike.
            # The short hold is deliberately limited so an old path is never
            # followed after the vehicle has moved significantly.
            waypoints = list(self.last_waypoints)
            self.path_evidence_count = min(self.path_evidence_count, 1)
        else:
            self.last_throttle = 0.0
            self.last_brake = 1.0
            self.last_steering = 0.0
            self.last_target_point = None
            self.path_started_at = None
            self.last_status = "STOP: no valid forward path"
            return self.last_throttle, self.last_steering, self.last_brake
        if self.path_started_at is None:
            self.path_started_at = time.monotonic()

        if self.path_evidence_count <= 1:
            target_speed = min(self.target_speed, 0.5)
        elif self.path_evidence_count < 4:
            target_speed = min(self.target_speed, 0.8)
        else:
            target_speed = self.target_speed

        closest_idx = 0
        for index, point in enumerate(waypoints):
            dx, dy = point[0] - vehicle_x, point[1] - vehicle_y
            if dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta) > 0.0:
                closest_idx = index
                break
        far_idx = closest_idx
        for index in range(closest_idx + 1, len(waypoints)):
            if self.get_distance(waypoints[index], waypoints[closest_idx]) >= 5.0:
                far_idx = index
                break
        if self.path_evidence_count >= 4 and far_idx > closest_idx and closest_idx + 1 < len(waypoints):
            far_yaw = math.atan2(
                waypoints[far_idx][1] - waypoints[closest_idx][1],
                waypoints[far_idx][0] - waypoints[closest_idx][0],
            )
            near_yaw = math.atan2(
                waypoints[closest_idx + 1][1] - waypoints[closest_idx][1],
                waypoints[closest_idx + 1][0] - waypoints[closest_idx][0],
            )
            yaw_difference = abs((far_yaw - near_yaw + math.pi) % (2 * math.pi) - math.pi)
            if yaw_difference > math.radians(20):
                target_speed = max(0.6, target_speed - 2.0 * (yaw_difference / math.radians(45)))

        throttle, brake = self.mpc_control(vehicle_speed, target_speed)
        if (vehicle_speed < PATH_MOVING_SPEED_MPS
                and time.monotonic() - self.path_started_at < PATH_START_DURATION_S):
            throttle, brake = max(throttle, PATH_START_THROTTLE), 0.0
        steering = self._slew_limit_steering(
            self.stanley_control(vehicle_x, vehicle_y, vehicle_theta, vehicle_speed, waypoints)
        )
        if self.path_turn_rejected:
            throttle, brake, steering = 0.0, 1.0, 0.0

        self.last_throttle = throttle
        self.last_brake = brake
        self.last_steering = steering
        if self.path_turn_rejected:
            self.last_status = "STOP: path would require excessive turn"
        elif self.using_learned_route:
            self.last_status = f"LEARNED TRACK: {self.path_evidence_count} route waypoints"
        elif self.path_evidence_count <= 1:
            self.last_status = "SEED PATH: 1 observed waypoint"
        elif self.path_evidence_count < 4:
            self.last_status = f"POLYLINE: {self.path_evidence_count} observed waypoints"
        else:
            self.last_status = f"TRACKING: {self.path_evidence_count} centreline waypoints"
        return throttle, steering, brake
