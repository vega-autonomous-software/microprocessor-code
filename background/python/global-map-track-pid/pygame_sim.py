"""
=============================================================
  FSA Racing Simulator
=============================================================
  This program simulates a Formula Student Autonomous car
  driving around a cone track.

  The car can either:
    - Drive itself (Autonomous mode) using a path planner + PID controller
    - Be controlled by you (Manual mode) using WASD keys

  Controls:
    W/S       → accelerate / brake  (manual)
    A/D       → steer left / right  (manual)
    ↑ / ↓     → raise / lower steering sensitivity  (autonomous)
    R         → reset car to start
    ESC       → quit
=============================================================
"""

import pygame
import math
from track_generator import build_cones
from Delunay_Triangulation import Path


# ==============================================================
#  CONSTANTS  (tweak these to change how the sim feels)
# ==============================================================

SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 720
FPS           = 60
PADDING       = 50       # pixels of empty space around the track edges

# Car physics
WHEELBASE     = 1.5      # distance between front and rear axle (metres)
MAX_ACCEL     = 3.0      # how fast the car can speed up (m/s²)
MAX_STEER_DEG = 30       # maximum steering angle in degrees

# What the car can "see" ahead of it
VIS_DEPTH     = 100   # how far forward the car can see (metres)
VIS_WIDTH     = 10   # half-width of the vision zone (metres)

# Pure Pursuit lookahead distance
LOOKAHEAD     = 4.0      # the car aims for a point this far ahead on the path

# PID gains — these three numbers control how the car steers automatically
#   Kp = how hard it reacts to an error RIGHT NOW
#   Ki = how much it corrects for an error that has been around a LONG TIME
#   Kd = how much it smooths out SUDDEN JERKY changes
KP_STEER_DEFAULT = 1.2
KI_STEER_DEFAULT = 0.0
KD_STEER_DEFAULT = 0.1

KP_SPEED = 1.5
KI_SPEED = 0.1
KD_SPEED = 0.05

TARGET_SPEED  = 4.0      # metres per second the car tries to reach


# ==============================================================
#  HELPER: tiny PID controller
# ==============================================================

class PID:
    """
    A PID controller — the brain behind smooth automatic control.

    Imagine you're in a shower and the water is too cold:
      P (proportional): you turn the knob A LOT because it's very cold right now
      I (integral):     you turn it a tiny extra bit because it's been cold a while
      D (derivative):   you slow your turning because it's warming up fast already

    Together, P + I + D get you to exactly the right temperature without
    overshooting into scalding hot water.
    """

    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit  # optional max/min on the output

        self._integral   = 0.0   # accumulated error over time
        self._prev_error = 0.0   # error from the previous frame

    def update(self, error: float, dt: float) -> float:
        """
        Feed in an error (how wrong things are), get back a correction.
        dt = time since last update (seconds).
        """
        if dt <= 0:
            return 0.0

        self._integral  += error * dt
        derivative       = (error - self._prev_error) / dt
        self._prev_error = error

        output = (self.kp * error
                + self.ki * self._integral
                + self.kd * derivative)

        # Clamp to allowed range if a limit was set
        if self.output_limit is not None:
            output = max(-self.output_limit, min(self.output_limit, output))

        return output

    def reset(self):
        """Wipe the memory — use this when restarting."""
        self._integral   = 0.0
        self._prev_error = 0.0


# ==============================================================
#  THE CAR
# ==============================================================

class Car:
    """
    A simple car model using the 'kinematic bicycle model'.

    Think of a bicycle: it has a front wheel that steers
    and a rear wheel that drives. Our car works the same way.

    State variables:
        x, y        → position on the track (metres)
        heading     → which direction the car is facing (radians)
        speed       → how fast it's going (m/s)
        steer_angle → how much the front wheel is turned (radians)
    """

    def __init__(self, x: float, y: float, heading: float):
        # Position and orientation
        self.x           = x
        self.y           = y
        self.heading     = heading

        # Motion state
        self.speed       = 0.0
        self.steer_angle = 0.0

        self.max_steer   = math.radians(MAX_STEER_DEG)

        # Two PID controllers: one steers, one manages speed
        self.steer_pid = PID(KP_STEER_DEFAULT, KI_STEER_DEFAULT, KD_STEER_DEFAULT,
                             output_limit=self.max_steer)
        self.speed_pid = PID(KP_SPEED, KI_SPEED, KD_SPEED, output_limit=1.0)

    # ----------------------------------------------------------
    #  Physics update  (called every frame)
    # ----------------------------------------------------------

    def update(self, dt: float):
        """
        Move the car forward by one tiny time step (1/60th of a second).

        Uses the bicycle model equations:
          new_x       = old_x + speed * cos(heading) * dt
          new_y       = old_y + speed * sin(heading) * dt
          new_heading = old_heading + (speed / wheelbase) * tan(steer) * dt
        """
        if self.speed < 0.001:
            return   # car is basically stopped — nothing to update

        self.x       += self.speed * math.cos(self.heading) * dt
        self.y       += self.speed * math.sin(self.heading) * dt

        # Steering changes the heading — sharp steer + high speed = big turn
        self.heading += (self.speed / WHEELBASE) * math.tan(self.steer_angle) * dt
        self.heading  = _wrap_angle(self.heading)

    # ----------------------------------------------------------
    #  Autonomous control
    # ----------------------------------------------------------

    def autonomous_control(self, waypoints: list, dt: float):
        """
        Let the car drive itself along a list of (x, y) waypoints.

        Three steps every frame:
          1. Pure Pursuit  → find a target point ahead on the path
          2. Steering PID  → smoothly steer toward that point
          3. Speed PID     → smoothly reach the target speed
        """
        if not waypoints:
            return

        # Step 1: Pure Pursuit — find the lookahead target
        target_x, target_y = self._find_lookahead(waypoints)

        # Angle from car to the target point
        angle_to_target = math.atan2(target_y - self.y, target_x - self.x)

        # alpha = angular error (how far left or right is the target?)
        alpha = _wrap_angle(angle_to_target - self.heading)

        # Pure pursuit formula: given alpha, compute the ideal steering angle
        desired_steer = math.atan2(2 * WHEELBASE * math.sin(alpha), LOOKAHEAD)

        # Step 2: Steering PID — don't snap to desired_steer instantly, ease into it
        steer_error      = desired_steer - self.steer_angle
        steer_correction = self.steer_pid.update(steer_error, dt)
        self.steer_angle = _clamp(
            self.steer_angle + steer_correction * dt,
            -self.max_steer, self.max_steer
        )

        # Step 3: Speed PID — push throttle to reach TARGET_SPEED
        speed_error  = TARGET_SPEED - self.speed
        throttle     = self.speed_pid.update(speed_error, dt)
        throttle     = _clamp(throttle, 0.0, 1.0)
        self.speed  += throttle * dt * MAX_ACCEL

    def _find_lookahead(self, waypoints: list):
        """
        Scan the path to find the first waypoint that is
        at least LOOKAHEAD metres away from the car.

        Like a runner looking a few metres ahead — not at their feet,
        not at the finish line, just a comfortable distance forward.
        """
        # First, find which waypoint we are closest to right now
        closest_idx = 0
        min_dist    = float('inf')
        for i, (px, py) in enumerate(waypoints):
            d = math.hypot(px - self.x, py - self.y)
            if d < min_dist:
                min_dist    = d
                closest_idx = i

        # Then walk forward from there until we're far enough ahead
        for i in range(len(waypoints)):
            idx    = (closest_idx + i) % len(waypoints)
            px, py = waypoints[idx]
            if math.hypot(px - self.x, py - self.y) >= LOOKAHEAD:
                return px, py

        return waypoints[closest_idx]   # fallback: just use closest

    # ----------------------------------------------------------
    #  Manual control
    # ----------------------------------------------------------

    def manual_control(self, keys, dt: float):
        """
        Let the player drive with the keyboard.

          W → speed up
          S → slow down / brake
          A → steer left
          D → steer right
        """
        # Steer left / right, or drift back to straight if no key pressed
        if keys[pygame.K_a]:
            self.steer_angle += math.radians(60) * dt    # turn left
        elif keys[pygame.K_d]:
            self.steer_angle -= math.radians(60) * dt    # turn right
        else:
            self.steer_angle *= 0.85   # return to centre smoothly

        self.steer_angle = _clamp(self.steer_angle, -self.max_steer, self.max_steer)

        # Throttle and brake
        if keys[pygame.K_w]:
            self.speed += 5.0 * dt
        if keys[pygame.K_s]:
            self.speed = max(0.0, self.speed - 5.0 * dt)

        # Natural friction — car slows down if you're not pressing W
        self.speed -= self.speed * 0.5 * dt
        self.speed  = max(0.0, self.speed)

    # ----------------------------------------------------------
    #  Sensor: which cones can the car see?
    # ----------------------------------------------------------

    def get_visible_cones(self, all_cones: list) -> list:
        """
        Return only the cones inside the car's forward vision zone.

        Imagine holding a flashlight while walking — you only see
        what's in the beam. The car's beam is a trapezoid:
        narrow right in front, wider further away.
        """
        visible = []
        for cone in all_cones:
            # Vector from car to cone
            dx = cone.x - self.x
            dy = cone.y - self.y

            # Rotate into the car's local frame so we can reason about
            # "ahead" and "to the side" separately
            local_fwd =  dx * math.cos(self.heading) + dy * math.sin(self.heading)
            local_lat = -dx * math.sin(self.heading) + dy * math.cos(self.heading)

            # Must be directly ahead (not behind the car)
            if not (0 <= local_fwd <= VIS_DEPTH):
                continue

            # The beam gets wider the further ahead you look
            allowed_width = VIS_WIDTH * (0.3 + 0.7 * (local_fwd / VIS_DEPTH))
            if abs(local_lat) <= allowed_width:
                visible.append(cone)

        return visible

    # ----------------------------------------------------------
    #  Cross-track error
    # ----------------------------------------------------------

    def cross_track_error(self, waypoints: list) -> float:
        """
        How far off the path is the car right now?
        0.0 = perfectly on the line. Bigger = further off.
        """
        if not waypoints:
            return 0.0
        return min(math.hypot(self.x - px, self.y - py) for px, py in waypoints)

    # ----------------------------------------------------------
    #  Reset
    # ----------------------------------------------------------

    def reset(self, x: float, y: float, heading: float):
        """Put the car back at the start position, completely fresh."""
        self.x           = x
        self.y           = y
        self.heading     = heading
        self.speed       = 0.0
        self.steer_angle = 0.0
        self.steer_pid.reset()
        self.speed_pid.reset()


# ==============================================================
#  LAP TIMER
# ==============================================================

class LapTimer:
    """
    Keeps track of how long each lap takes.

    Works like a stopwatch:
      - Starts when the car leaves the start line
      - Resets every time the car crosses it again
    """

    def __init__(self, start_x: float, start_y: float):
        self.start_x = start_x
        self.start_y = start_y

        self._lap_start    = pygame.time.get_ticks() / 1000.0
        self._distance     = 0.0              # metres driven since lap started
        self._last_pos     = (start_x, start_y)
        self.lap_count     = 0
        self.last_lap_time = None

    def update(self, car_x: float, car_y: float) -> bool:
        """
        Call every frame. Returns True the moment a lap is completed.

        We only count a lap if the car has driven at least 10 metres —
        this stops it from triggering immediately at the very start.
        """
        # Measure how far the car moved since last frame
        step = math.hypot(car_x - self._last_pos[0], car_y - self._last_pos[1])
        self._distance += step
        self._last_pos  = (car_x, car_y)

        near_start    = math.hypot(car_x - self.start_x, car_y - self.start_y) < 2.0
        driven_enough = self._distance > 10.0

        if near_start and driven_enough:
            # A lap is complete!
            now = pygame.time.get_ticks() / 1000.0
            self.last_lap_time = now - self._lap_start
            self.lap_count    += 1
            print(f"  ✓ Lap {self.lap_count}: {self.last_lap_time:.2f}s")

            # Reset for the next lap
            self._lap_start = now
            self._distance  = 0.0
            return True

        return False

    def current_lap_time(self) -> float:
        """Seconds elapsed since this lap started."""
        return pygame.time.get_ticks() / 1000.0 - self._lap_start

    def reset(self):
        """Restart the timer from scratch."""
        self._lap_start = pygame.time.get_ticks() / 1000.0
        self._distance  = 0.0
        self._last_pos  = (self.start_x, self.start_y)


# ==============================================================
#  RENDERER  (draws everything on screen)
# ==============================================================

class Renderer:
    """
    Handles all drawing — cones, paths, car, vision zone, and HUD.

    The key job of this class is converting real-world metres
    into pixel coordinates on screen.
    """

    def __init__(self, screen, all_cones: list):
        self.screen = screen
        self.font   = pygame.font.SysFont(None, 24)

        # Auto-scale: fit the whole track into the window with padding
        all_x = [c.x for c in all_cones]
        all_y = [c.y for c in all_cones]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        scale_x    = (SCREEN_WIDTH  - 2 * PADDING) / (max_x - min_x)
        scale_y    = (SCREEN_HEIGHT - 2 * PADDING) / (max_y - min_y)
        self.scale  = min(scale_x, scale_y)

        # Offset so the track is centred on screen
        self.offset_x = SCREEN_WIDTH  / 2 - (min_x + max_x) / 2 * self.scale
        self.offset_y = SCREEN_HEIGHT / 2 - (min_y + max_y) / 2 * self.scale

    def to_screen(self, x: float, y: float):
        """
        Convert world coordinates (metres) → screen pixels.
        Note: screen Y goes DOWN, but world Y goes UP, so we flip it.
        """
        sx = int(x * self.scale + self.offset_x)
        sy = int(SCREEN_HEIGHT - (y * self.scale + self.offset_y))
        return (sx, sy)

    # ----------------------------------------------------------

    def draw_background(self):
        """Fill the screen with dark grey — like looking down at tarmac."""
        self.screen.fill((20, 20, 20))

    def draw_cones(self, cones: list):
        """
        Draw each cone as a coloured circle.
          Blue   = left boundary of the track
          Yellow = right boundary of the track
          Orange = start / finish line markers
        """
        colour_map = {
            "blue":         ( 50, 100, 255),
            "yellow":       (255, 220,   0),
            "small_orange": (255, 140,   0),
            "large_orange": (255, 100,   0),
        }
        radius_map = {
            "blue": 5, "yellow": 5,
            "small_orange": 8, "large_orange": 10,
        }
        for cone in cones:
            sx, sy = self.to_screen(cone.x, cone.y)
            colour = colour_map.get(cone.cone_type, (200, 200, 200))
            radius = radius_map.get(cone.cone_type, 5)
            pygame.draw.circle(self.screen, colour, (sx, sy), radius)

    def draw_path(self, waypoints: list, colour, width: int = 2):
        """Draw a smooth polyline through the given (x, y) waypoints."""
        if len(waypoints) < 2:
            return
        pts = [self.to_screen(x, y) for x, y in waypoints]
        pygame.draw.lines(self.screen, colour, False, pts, width)

    def draw_visibility_zone(self, car: Car):
        """
        Draw the trapezoidal 'flashlight beam' in front of the car.
        Green zone = what the car can currently see.
        Narrow near the car, wider further ahead.
        """
        # Define the four corners in car-local space: (forward_dist, side_offset)
        # Think of it as: (how far ahead, how far left/right)
        corners_local = [
            (VIS_DEPTH,  VIS_WIDTH * 5),   # front-left  (far, wide)
            (VIS_DEPTH, -VIS_WIDTH * 5),   # front-right (far, wide)
            (0,         -VIS_WIDTH * 0.2),   # rear-right  (close, narrow)
            (0,          VIS_WIDTH * 0.2),   # rear-left   (close, narrow)
        ]

        # Rotate each local corner into world space using the car's heading
        screen_pts = []
        for fwd, lat in corners_local:
            wx = car.x + fwd * math.cos(car.heading) - lat * math.sin(car.heading)
            wy = car.y + fwd * math.sin(car.heading) + lat * math.cos(car.heading)
            screen_pts.append(self.to_screen(wx, wy))

        # Draw on a transparent surface so it blends with the background
        vis_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(vis_surface, (0, 200,   0,  55), screen_pts)    # fill
        pygame.draw.polygon(vis_surface, (0, 255,   0, 180), screen_pts, 2) # border
        self.screen.blit(vis_surface, (0, 0))

    def draw_car(self, car: Car):
        """
        Draw the car as a small red triangle.
        The pointy end = the front of the car (where it's heading).
        """
        size = 0.8   # world-unit size of the triangle

        # Three corners of the triangle in world space
        front = (car.x + size * math.cos(car.heading),
                 car.y + size * math.sin(car.heading))
        left  = (car.x + size * math.cos(car.heading + 2.4),
                 car.y + size * math.sin(car.heading + 2.4))
        right = (car.x + size * math.cos(car.heading - 2.4),
                 car.y + size * math.sin(car.heading - 2.4))

        pts = [self.to_screen(*front),
               self.to_screen(*left),
               self.to_screen(*right)]

        pygame.draw.polygon(self.screen, (220,  30,  30), pts)      # dark red fill
        pygame.draw.polygon(self.screen, (255, 120, 120), pts, 2)   # lighter border

    def draw_hud(self, car: Car, lap_timer: LapTimer,
                 is_autonomous: bool, first_lap: bool, cte: float):
        """
        Draw the Heads-Up Display in the top-left corner.
        Shows speed, steering, cross-track error, lap time, and PID info.
        """
        if is_autonomous:
            phase = "Centerline (lap 1)" if first_lap else "Racing Line (lap 2+)"
            mode_label = f"Autonomous — {phase}"
        else:
            mode_label = "Manual  |  WASD to drive"

        lines = [
            f"Mode:              {mode_label}",
            f"Speed:             {car.speed:.2f} m/s",
            f"Steering:          {math.degrees(car.steer_angle):.1f}°",
            f"Cross-track error: {cte:.2f} m",
            f"Lap time:          {lap_timer.current_lap_time():.2f} s",
            f"Laps completed:    {lap_timer.lap_count}",
        ]

        if is_autonomous:
            lines += [
                "",
                f"KP steer: {car.steer_pid.kp:.1f}   (↑ / ↓ to change)",
                f"KI steer: {car.steer_pid.ki:.2f}",
                f"KD steer: {car.steer_pid.kd:.2f}",
            ]
        else:
            lines.append("R = reset   ESC = quit")

        for i, line in enumerate(lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 24))


# ==============================================================
#  MAIN SIMULATOR
# ==============================================================

class Simulator:
    """
    The top-level class that runs the whole simulation.

    It owns everything:
      - the track (cones)
      - the car
      - the path planner
      - the renderer
      - the lap timer
      - the main game loop
    """

    def __init__(self, is_autonomous: bool):
        self.is_autonomous = is_autonomous

        # Load all the track cones
        self.all_cones  = build_cones()
        self.seen_cones = []   # only the cones the car has spotted so far

        # Path data
        self.current_path = []   # path built from seen cones (updated every frame)
        self.racing_line  = []   # full optimised racing line (computed after lap 1)
        self.first_lap    = True # are we still on the first lap?

        # Work out where the car should start
        start_x, start_y, start_heading = self._find_start_pose()
        self._start_x       = start_x
        self._start_y       = start_y
        self._start_heading = start_heading

        # Create the core objects
        self.car       = Car(start_x, start_y, start_heading)
        self.renderer  = Renderer(screen, self.all_cones)
        self.lap_timer = LapTimer(start_x, start_y)

    # ----------------------------------------------------------
    #  Start position
    # ----------------------------------------------------------

    def _find_start_pose(self):
        """
        Place the car between the two orange cones at the start/finish line,
        facing along the track toward the first blue cone.
        """
        orange = [c for c in self.all_cones
                  if c.cone_type in ("small_orange", "large_orange")]

        if len(orange) >= 2:
            # Start exactly halfway between the two orange cones
            start_x = (orange[0].x + orange[1].x) / 2
            start_y = (orange[0].y + orange[1].y) / 2
        else:
            start_x, start_y = self.all_cones[0].x, self.all_cones[0].y

        # Face toward the first blue cone (that's the direction of travel)
        first_blue = next((c for c in self.all_cones if c.cone_type == "blue"), None)
        if first_blue:
            heading = math.atan2(first_blue.y - start_y, first_blue.x - start_x)
        else:
            heading = 0.0

        return start_x, start_y, heading

    # ----------------------------------------------------------
    #  Path planning
    # ----------------------------------------------------------

    def _rebuild_path_from_seen_cones(self) -> list:
        """
        Ask the path planner to build a centerline from the cones
        the car has discovered so far.

        Returns an empty list if we haven't seen enough cones yet
        (Delaunay triangulation needs at least 4 points).
        """
        if len(self.seen_cones) < 4:
            return []
        try:
            planner = Path(self.seen_cones)
            return planner.get_centerline()
        except Exception as e:
            print(f"[Path] Planning failed: {e}")
            return []

    def _compute_full_racing_line(self):
        """
        After lap 1 is done, compute the optimal racing line using
        the full cone map. This solves an optimisation problem —
        it takes a moment but only runs once.
        """
        print("[Path] Computing full racing line — please wait...")
        planner          = Path(self.all_cones)
        self.racing_line = planner.get_racing_line()
        print(f"[Path] Racing line ready: {len(self.racing_line)} waypoints")

    # ----------------------------------------------------------
    #  Input handling
    # ----------------------------------------------------------

    def _handle_keydown(self, key) -> bool:
        """
        React to a single key press.
        Returns False if the game should quit, True to keep running.
        """
        if key == pygame.K_ESCAPE:
            return False   # quit the game

        elif key == pygame.K_r:
            self._reset()

        # Live PID tuning — change steering aggressiveness on the fly
        elif self.is_autonomous and key == pygame.K_UP:
            self.car.steer_pid.kp = round(self.car.steer_pid.kp + 0.1, 2)
            print(f"[PID] KP_STEER → {self.car.steer_pid.kp:.1f}")

        elif self.is_autonomous and key == pygame.K_DOWN:
            self.car.steer_pid.kp = round(self.car.steer_pid.kp - 0.1, 2)
            print(f"[PID] KP_STEER → {self.car.steer_pid.kp:.1f}")

        return True

    def _reset(self):
        """Put everything back to the very beginning."""
        self.car.reset(self._start_x, self._start_y, self._start_heading)
        self.seen_cones   = []
        self.current_path = []
        self.racing_line  = []
        self.first_lap    = True
        self.lap_timer.reset()
        print("[Reset] Car reset to start.")

    # ----------------------------------------------------------
    #  Game loop
    # ----------------------------------------------------------

    def run(self):
        """
        The main loop — this runs 60 times per second.

        Each tick (frame) does exactly these steps in order:
          1.  Handle keyboard / window events
          2.  Update which cones the car can see
          3.  Rebuild / select the path to follow
          4.  Control the car (auto or manual)
          5.  Step physics forward
          6.  Check if a lap was just completed
          7.  Draw everything on screen
        """
        running = True

        while running:
            # How many seconds have passed since the last frame?
            dt = clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)   # cap at 50 ms — avoids physics blow-up on a lag spike

            # ── 1. Events ────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keydown(event.key)

            # ── 2. Update visible cones ───────────────────────────────
            # The car "discovers" cones as it drives near them
            for cone in self.car.get_visible_cones(self.all_cones):
                if cone not in self.seen_cones:
                    self.seen_cones.append(cone)

            # ── 3. Choose which path to follow ────────────────────────
            if self.is_autonomous:
                if self.first_lap:
                    # Lap 1: build path from whatever we've seen so far
                    self.current_path = self._rebuild_path_from_seen_cones()
                    follow = self.current_path
                else:
                    # Lap 2+: use the full optimised racing line
                    if not self.racing_line:
                        self._compute_full_racing_line()
                    follow = self.racing_line
            else:
                # Manual mode: still show a path for reference, but don't follow it
                self.current_path = self._rebuild_path_from_seen_cones()
                follow = self.current_path

            # ── 4. Car control ────────────────────────────────────────
            if self.is_autonomous:
                self.car.autonomous_control(follow, dt)
            else:
                self.car.manual_control(pygame.key.get_pressed(), dt)

            # ── 5. Physics ────────────────────────────────────────────
            self.car.update(dt)

            # ── 6. Lap tracking ───────────────────────────────────────
            lap_done = self.lap_timer.update(self.car.x, self.car.y)
            if lap_done and self.first_lap and self.is_autonomous:
                print("[Lap] First lap complete — switching to racing line!")
                self.first_lap = False

            # ── 7. Draw ───────────────────────────────────────────────
            cte = self.car.cross_track_error(follow)
            self._draw(follow, cte)

        pygame.quit()

    # ----------------------------------------------------------
    #  Drawing
    # ----------------------------------------------------------

    def _draw(self, follow_path: list, cte: float):
        """Render one complete frame."""
        self.renderer.draw_background()
        self.renderer.draw_visibility_zone(self.car)
        self.renderer.draw_cones(self.seen_cones)

        if follow_path:
            # Green = lap 1 centerline, White = lap 2+ racing line
            colour = (0, 220, 100) if self.first_lap else (255, 255, 255)
            self.renderer.draw_path(follow_path, colour, width=2)

        self.renderer.draw_car(self.car)
        self.renderer.draw_hud(
            self.car, self.lap_timer,
            self.is_autonomous, self.first_lap, cte
        )

        pygame.display.flip()


# ==============================================================
#  SMALL UTILITY FUNCTIONS
# ==============================================================

def _wrap_angle(angle: float) -> float:
    """
    Keep an angle between -π and +π (i.e. -180° to +180°).
    Like making sure a compass reading never goes past 360°.
    """
    while angle >  math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle


def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Force a value to stay between lo and hi.
    Like making sure a volume knob stays between 0 and 100.
    """
    return max(lo, min(hi, value))


# ==============================================================
#  ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    print("=" * 42)
    print("   FSA Racing Simulator")
    print("=" * 42)
    print("   1 → Autonomous  (car drives itself)")
    print("   2 → Manual      (you drive with WASD)")
    choice    = input("   Choose [1/2]: ").strip()
    is_auto   = (choice == "1")

    # Start pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("FSA Racing Simulator")
    clock  = pygame.time.Clock()

    # Fire it up!
    Simulator(is_autonomous=is_auto).run()