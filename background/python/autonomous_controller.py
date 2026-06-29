import math
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

class AutonomousController:
    def __init__(self, target_speed=4.0):
        # Kinematic Bicycle Model Constraints
        self.L = 1.53  # Wheelbase (m)
        self.max_steer_angle = math.radians(25.0)
        self.max_acceleration = 5.0  # m/s^2
        self.max_deceleration = 8.0  # m/s^2
        
        self.target_speed = target_speed
        
        # State tracking
        self.last_target_point = None
        self.last_waypoints = []
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_steering = 0.0

    def get_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def extract_centerline(self, ekf_state, vehicle_x=0.0, vehicle_y=0.0, vehicle_theta=0.0, max_track_width=8.0):
        """
        Uses Delaunay Triangulation to extract the centerline between blue and yellow cones.
        """
        blue_cones = [c for c in ekf_state if c['color'] == 'blue']
        yellow_cones = [c for c in ekf_state if c['color'] == 'yellow']
        all_cones = blue_cones + yellow_cones
        
        # Local Horizon Fix: Only triangulate cones in front of the car
        cones = []
        for c in all_cones:
            dx = c['x'] - vehicle_x
            dy = c['y'] - vehicle_y
            local_x = dx * math.cos(vehicle_theta) + dy * math.sin(vehicle_theta)
            if local_x > -1.0 and math.hypot(dx, dy) < 15.0:
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
            except Exception as e:
                pass

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
                        
                if best_pt is None or best_score > 10.0:
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
                    tck, u = splprep([pts[:, 0], pts[:, 1]], s=3.0, k=3)
                    u_new = np.linspace(0, 1, len(pts) * 3)
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
            throttle = opt_accel / self.max_acceleration
        else:
            brake = -opt_accel / self.max_deceleration
            
        return throttle, brake

    def stanley_control(self, vehicle_x, vehicle_y, vehicle_theta, vehicle_speed, waypoints):
        """
        Stanley Controller for Lateral Control.
        """
        if not waypoints:
            return self.last_steering
            
        # IMPORTANT: FSDS IMU Yaw is inverted compared to standard Atan2 geometry!
        # Convert to Standard Geometric Yaw (CCW positive from +X)
        geom_theta = math.pi - vehicle_theta
        
        closest_idx = None
        target_local_y = None
        
        for i, pt in enumerate(waypoints):
            dx = pt[0] - vehicle_x
            dy = pt[1] - vehicle_y
            
            # Standard 2D Rotation Matrix using geom_theta
            local_x = dx * math.cos(geom_theta) + dy * math.sin(geom_theta)
            local_y = -dx * math.sin(geom_theta) + dy * math.cos(geom_theta)
            
            # ONLY consider waypoints that are IN FRONT of the car
            if local_x > 0.0:
                closest_idx = i
                target_local_y = local_y
                break
                
        if closest_idx is None:
            return self.last_steering
            
        target_wp = waypoints[closest_idx]
        self.last_target_point = (target_wp[0], target_wp[1])
        
        e_f = target_local_y
        
        # Path Heading Error (psi_e)
        # Look ahead geometrically (approx 1.5 meters) to avoid B-Spline noise
        lookahead_idx = closest_idx
        for j in range(closest_idx + 1, len(waypoints)):
            dist_to_j = math.hypot(waypoints[j][0] - target_wp[0], waypoints[j][1] - target_wp[1])
            if dist_to_j >= 1.5:
                lookahead_idx = j
                break
                
        if lookahead_idx > closest_idx:
            next_wp = waypoints[lookahead_idx]
            path_yaw = math.atan2(next_wp[1] - target_wp[1], next_wp[0] - target_wp[0])
        else:
            path_yaw = geom_theta
            
        psi_e = path_yaw - geom_theta
        psi_e = (psi_e + math.pi) % (2 * math.pi) - math.pi
        
        k_e = 1.0 # Cross-track error gain
        k_s = 1.0 # Softening constant
        
        safe_speed = max(1.0, vehicle_speed)
        steering = psi_e + math.atan2(k_e * e_f, safe_speed + k_s)
        
        steering = max(-1.0, min(1.0, steering / 0.5))
        return steering

    def compute_commands(self, ekf_state, vehicle_x, vehicle_y, vehicle_theta, vehicle_speed):
        waypoints = self.extract_centerline(ekf_state, vehicle_x, vehicle_y, vehicle_theta)
        
        if not waypoints:
            self.last_throttle = 0.0
            self.last_brake = min(1.0, self.last_brake + 0.05)
            return self.last_throttle, self.last_steering, self.last_brake
            
        # 1. Target Speed Profile Generation
        target_speed = self.target_speed
        
        # Estimate upcoming curvature
        closest_idx = 0
        geom_theta = math.pi - vehicle_theta
        for i, pt in enumerate(waypoints):
            dx = pt[0] - vehicle_x
            dy = pt[1] - vehicle_y
            local_x = dx * math.cos(geom_theta) + dy * math.sin(geom_theta)
            if local_x > 0.0:
                closest_idx = i
                break
                
        far_idx = closest_idx
        for j in range(closest_idx + 1, len(waypoints)):
            dist = math.hypot(waypoints[j][0] - waypoints[closest_idx][0], waypoints[j][1] - waypoints[closest_idx][1])
            if dist >= 5.0:
                far_idx = j
                break
                
        if far_idx > closest_idx and closest_idx + 1 < len(waypoints):
            dy = waypoints[far_idx][1] - waypoints[closest_idx][1]
            dx = waypoints[far_idx][0] - waypoints[closest_idx][0]
            far_yaw = math.atan2(dy, dx)
            
            near_dy = waypoints[closest_idx+1][1] - waypoints[closest_idx][1]
            near_dx = waypoints[closest_idx+1][0] - waypoints[closest_idx][0]
            near_yaw = math.atan2(near_dy, near_dx)
            
            yaw_diff = abs((far_yaw - near_yaw + math.pi) % (2*math.pi) - math.pi)
            
            # Slow down proportionally for corners
            if yaw_diff > math.radians(20):
                target_speed = max(2.0, self.target_speed - 2.0 * (yaw_diff / math.radians(45)))
                
        # 2. Longitudinal Control (MPC)
        throttle, brake = self.mpc_control(vehicle_speed, target_speed)
        
        # 3. Lateral Control (Stanley)
        steering = self.stanley_control(vehicle_x, vehicle_y, vehicle_theta, vehicle_speed, waypoints)
        
        self.last_throttle = throttle
        self.last_brake = brake
        self.last_steering = steering
        
        return throttle, steering, brake
