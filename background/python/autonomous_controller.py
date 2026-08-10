import math
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

class AutonomousController:
    def __init__(self, target_speed=8.0):
        # Kinematic Bicycle Model Constraints
        self.L = 1.53  # Wheelbase (m)
        self.max_steer_angle = math.radians(45.0)
        self.max_acceleration = 8.0  # m/s^2
        self.max_deceleration = 11.0  # m/s^2
        
        # Pure Pursuit Tuning Parameters
        self.min_lookahead = 1.5  # Minimum lookahead distance (m)
        self.max_lookahead = 6.0  # Maximum lookahead distance (m)
        self.lookahead_speed_gain = 0.35  # Speed scaling factor for lookahead distance
        
        self.target_speed = target_speed
        self.filtered_target_speed = target_speed
        
        # State tracking
        self.last_target_point = None
        self.last_waypoints = []
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_steering = 0.0

    def get_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def extract_centerline(self, ekf_state, vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0, max_track_width=20.0):
        """
        Extracts the track centerline between blue (left boundary) and yellow (right boundary) cones.
        """
        blue_cones = [c for c in ekf_state if 'blue' in c['color']]
        yellow_cones = [c for c in ekf_state if 'yellow' in c['color']]
        all_cones = blue_cones + yellow_cones
        
        # Local Horizon Filter: Only consider cones in front / near the vehicle
        cones = []
        for c in all_cones:
            dx = c['x'] - vehicle_x
            dy = c['y'] - vehicle_y
            local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
            if local_x > -2.0 and math.hypot(dx, dy) < 30.0:
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
                
                    # Vector from Blue (left) to Yellow (right)
                    v_dx = p_yellow[0] - p_blue[0]
                    v_dy = p_yellow[1] - p_blue[1]
            
                    # Rotate 90 deg CCW (from Right to Forward track direction):
                    # (dx, dy) rotated 90 deg CCW is (-dy, dx)
                    fwd_dx = -v_dy
                    fwd_dy = v_dx
            
                    mag = math.hypot(fwd_dx, fwd_dy)
                    if mag > 0.001:
                        fwd_dx /= mag
                        fwd_dy /= mag
                        waypoints.append((mx, my, fwd_dx, fwd_dy))
            except Exception as e:
                pass

        # Pairwise matching fallback if Delaunay missed sparse cone pairs
        if len(waypoints) < 2 and blue_cones and yellow_cones:
            for b in blue_cones:
                for y in yellow_cones:
                    dist = math.hypot(y['x'] - b['x'], y['y'] - b['y'])
                    if dist <= max_track_width:
                        mx = (b['x'] + y['x']) / 2.0
                        my = (b['y'] + y['y']) / 2.0
                        v_dx = y['x'] - b['x']
                        v_dy = y['y'] - b['y']
                        fwd_dx = -v_dy
                        fwd_dy = v_dx
                        mag = math.hypot(fwd_dx, fwd_dy)
                        if mag > 0.001:
                            fwd_dx /= mag
                            fwd_dy /= mag
                            waypoints.append((mx, my, fwd_dx, fwd_dy))

        if not waypoints:
            return []

        # Directional Nearest Neighbor Waypoint Sorting
        ordered_waypoints = []
        if waypoints:
            # Filter waypoints to those ahead or near car (local_x >= -2.0)
            fwd_wps = []
            for wp in waypoints:
                dx = wp[0] - vehicle_x
                dy = wp[1] - vehicle_y
                local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
                if local_x >= -2.0:
                    fwd_wps.append(wp)
            
            candidate_wps = fwd_wps if fwd_wps else waypoints
            
            # Start from waypoint closest to vehicle
            current_pt = min(candidate_wps, key=lambda p: math.hypot(p[0] - vehicle_x, p[1] - vehicle_y))
            ordered_waypoints.append(current_pt)
            unvisited = set(waypoints)
            if current_pt in unvisited:
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
                    if dot < -0.2:  # Align with forward movement direction
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

        # B-Spline Smoothing
        if len(ordered_waypoints) >= 4:
            try:
                pts = np.array([(p[0], p[1]) for p in ordered_waypoints])
                _, idx = np.unique(pts, axis=0, return_index=True)
                pts = pts[np.sort(idx)]
                
                if len(pts) >= 4:
                    tck, u = splprep([pts[:, 0], pts[:, 1]], s=2.0, k=min(3, len(pts)-1))
                    u_new = np.linspace(0, 1, max(len(pts) * 3, 20))
                    new_points = splev(u_new, tck)
                    ordered_waypoints = list(zip(new_points[0], new_points[1]))
                else:
                    ordered_waypoints = [(p[0], p[1]) for p in ordered_waypoints]
            except Exception as e:
                ordered_waypoints = [(p[0], p[1]) for p in ordered_waypoints]
        else:
            ordered_waypoints = [(p[0], p[1]) for p in ordered_waypoints]

        self.last_waypoints = ordered_waypoints
        return ordered_waypoints

    def mpc_control(self, current_speed, target_speed):
        """
        Model Predictive Controller (MPC) for Longitudinal Speed tracking.
        Optimizes a sequence of accelerations to match target speed while minimizing jerk.
        """
        N = 5       # Prediction horizon
        dt = 0.2    # Time step (1.0 second lookahead total)
        
        a0 = np.zeros(N)
        bounds = [(-self.max_deceleration, self.max_acceleration) for _ in range(N)]
        
        def cost_fn(a):
            cost = 0.0
            v = current_speed
            for k in range(N):
                v = v + a[k] * dt
                # Penalize deviation from target speed
                cost += 10.0 * (v - target_speed)**2
                # Penalize control effort
                cost += 1.0 * (a[k])**2
                # Penalize jerk
                if k > 0:
                    cost += 5.0 * (a[k] - a[k-1])**2
            return cost
            
        try:
            res = minimize(cost_fn, a0, bounds=bounds, method='SLSQP')
            if res.success:
                opt_accel = res.x[0]
            else:
                opt_accel = 1.5 * (target_speed - current_speed)
        except Exception:
            opt_accel = 1.5 * (target_speed - current_speed)
            
        opt_accel = max(-self.max_deceleration, min(self.max_acceleration, opt_accel))
            
        throttle = 0.0
        brake = 0.0
        
        if opt_accel > 0:
            throttle = min(1.0, opt_accel / self.max_acceleration)
        else:
            speed_error = max(0.0, current_speed - target_speed)
            brake_from_accel = -opt_accel / self.max_deceleration
            brake_from_speed_err = speed_error / 5.0
            brake = min(1.0, max(brake_from_accel, brake_from_speed_err))
            
        if brake > 0.02:
            throttle = 0.0

        return throttle, brake

    def pure_pursuit_control(self, vehicle_x, vehicle_y, vehicle_theta, vehicle_speed, waypoints):
        """
        Pure Pursuit Controller for Lateral Control and Line Following.
        Returns tuple of (steering, cross_track_error).
        """
        if not waypoints:
            return self.last_steering, 0.0

        # Calculate dynamic lookahead distance L_d based on vehicle speed
        lookahead_dist = max(
            self.min_lookahead,
            min(self.max_lookahead, self.min_lookahead + self.lookahead_speed_gain * vehicle_speed)
        )

        # Find waypoints in front of the vehicle
        forward_candidates = []
        for i, pt in enumerate(waypoints):
            dx = pt[0] - vehicle_x
            dy = pt[1] - vehicle_y

            local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
            local_y = -dx * math.sin(vehicle_theta) + dy * math.cos(vehicle_theta)
            dist = math.hypot(local_x, local_y)

            if local_x > 0.0:  # Strictly in front of vehicle
                forward_candidates.append((i, pt, local_x, local_y, dist))

        if not forward_candidates:
            # Fallback: if no point is ahead, pick the last waypoint
            target_pt = waypoints[-1]
            dx = target_pt[0] - vehicle_x
            dy = target_pt[1] - vehicle_y
            target_local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
            target_local_y = -dx * math.sin(vehicle_theta) + dy * math.cos(vehicle_theta)
            dist = max(0.5, math.hypot(target_local_x, target_local_y))
        else:
            # Search for candidate waypoint closest to lookahead distance L_d
            best_cand = None
            best_diff = float('inf')
            
            for cand in forward_candidates:
                diff = abs(cand[4] - lookahead_dist)
                if diff < best_diff:
                    best_diff = diff
                    best_cand = cand
            
            target_local_x = best_cand[2]
            target_local_y = best_cand[3]
            dist = best_cand[4]
            target_pt = best_cand[1]

        self.last_target_point = (target_pt[0], target_pt[1])
        cross_track_error = target_local_y

        # Pure Pursuit geometric curvature calculation: kappa = 2 * y_local / (d^2)
        if dist > 0.1:
            raw_steering = -math.atan2(2.0 * self.L * target_local_y, dist**2)
        else:
            raw_steering = 0.0

        # Normalize steering to [-1.0, 1.0] based on max steering angle (45 degrees)
        steering = max(-1.0, min(1.0, raw_steering / self.max_steer_angle))
        return steering, cross_track_error

    def compute_commands(self, ekf_state, vehicle_x, vehicle_y, vehicle_theta, vehicle_speed):
        waypoints = self.extract_centerline(ekf_state, vehicle_x, vehicle_y, vehicle_theta)
        
        if not waypoints:
            # If search / startup (speed low), crawl forward to find cones
            if vehicle_speed < 0.5:
                raw_throttle = 0.2
                raw_brake = 0.0
                raw_steering = 0.0
            else:
                raw_throttle = 0.0
                raw_brake = 0.3
                raw_steering = 0.0
            
            # Smooth command transition even when no waypoints are detected
            throttle = self.last_throttle * 0.70 + raw_throttle * 0.30
            brake = self.last_brake * 0.65 + raw_brake * 0.35
            steering = self.last_steering * 0.70 + raw_steering * 0.30
            
            self.last_throttle = throttle
            self.last_brake = brake
            self.last_steering = steering
            return throttle, steering, brake
            
        # 1. Compute Lateral Control (Pure Pursuit) to get steering demand and cross-track error
        steering, e_f = self.pure_pursuit_control(vehicle_x, vehicle_y, vehicle_theta, vehicle_speed, waypoints)

        # 2. Target Speed Profile Generation
        base_target_speed = self.target_speed
        
        # Estimate upcoming curvature across multiple lookahead horizons (2.0m, 4.0m, 6.0m)
        closest_idx = 0
        for i, pt in enumerate(waypoints):
            dx = pt[0] - vehicle_x
            dy = pt[1] - vehicle_y
            local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
            if local_x > 0.0:
                closest_idx = i
                break
                
        max_yaw_diff = 0.0
        if closest_idx < len(waypoints) - 1:
            near_pt = waypoints[closest_idx]
            for lookahead_m in [2.0, 4.0, 6.0]:
                far_idx = closest_idx
                for j in range(closest_idx + 1, len(waypoints)):
                    dist = math.hypot(waypoints[j][0] - near_pt[0], waypoints[j][1] - near_pt[1])
                    if dist >= lookahead_m:
                        far_idx = j
                        break
                if far_idx > closest_idx + 1:
                    dy = waypoints[far_idx][1] - waypoints[closest_idx][1]
                    dx = waypoints[far_idx][0] - waypoints[closest_idx][0]
                    far_yaw = math.atan2(dy, dx)
                    
                    near_dy = waypoints[closest_idx+1][1] - waypoints[closest_idx][1]
                    near_dx = waypoints[closest_idx+1][0] - waypoints[closest_idx][0]
                    near_yaw = math.atan2(near_dy, near_dx)
                    
                    ydiff = abs((far_yaw - near_yaw + math.pi) % (2*math.pi) - math.pi)
                    if ydiff > max_yaw_diff:
                        max_yaw_diff = ydiff
            
        # Target Speed rule: Continuous & smooth derating for steering angle & curve
        steer_mag = abs(steering)
        steer_factor = min(1.0, max(steer_mag / 0.6, max_yaw_diff / math.radians(40)))
        cte_factor = min(1.0, abs(e_f) / 1.2)
        combined_derate = max(steer_factor, cte_factor)
        
        # Smoothly derate target speed down to a minimum floor of 2.0 m/s
        raw_target_speed = max(2.0, base_target_speed * (1.0 - 0.7 * combined_derate))

        # Filter target speed over time (EMA low-pass) to prevent target speed step jumps
        self.filtered_target_speed = self.filtered_target_speed * 0.8 + raw_target_speed * 0.2
        target_speed = self.filtered_target_speed

        # 3. Longitudinal Control (MPC)
        throttle, brake = self.mpc_control(vehicle_speed, target_speed)
        
        # 4. Cornering & Steering Speed Control: Smooth active braking when steering at speed
        if steer_mag > 0.08 and vehicle_speed > 2.0:
            speed_excess = vehicle_speed - 2.0
            turn_intensity = min(1.0, (steer_mag - 0.08) / 0.5)
            turn_brake = min(0.5, turn_intensity * (0.10 + 0.15 * speed_excess))
            brake = max(brake, turn_brake)

        # 5. Actuator Command Low-Pass Filtering (Exponential Moving Average)
        # Prevents high frequency chatter and sudden brake/throttle spikes
        throttle = self.last_throttle * 0.70 + throttle * 0.30
        brake = self.last_brake * 0.65 + brake * 0.35
        
        # Smooth cross-fade between brake and throttle
        if brake > 0.02:
            throttle = throttle * max(0.0, 1.0 - (brake / 0.1))

        self.last_throttle = throttle
        self.last_brake = brake
        self.last_steering = steering
        
        return throttle, steering, brake

