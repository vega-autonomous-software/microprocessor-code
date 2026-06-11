import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from typing import List, Tuple, Set
from track_generator import Cone, build_cones
from dataclasses import dataclass


# Constants
LEFT_TYPES = {"blue", "small_orange", "large_orange"}
RIGHT_TYPES = {"yellow", "small_orange", "large_orange"}
CENTERLINE_LEFT_TYPES = {"blue"}
CENTERLINE_RIGHT_TYPES = {"yellow"}


@dataclass
class Triangle:
    """Represents a triangle from Delaunay triangulation with three cone indices."""
    indices: Tuple[int, int, int]

    def __post_init__(self):
        self.indices = tuple(sorted(self.indices))

    def get_cone_types(self, cones: List[Cone]) -> List[str]:
        return [cones[i].cone_type for i in self.indices]

    def has_mixed_types(self, cones: List[Cone]) -> bool:
        types = set(self.get_cone_types(cones))
        return bool(types & LEFT_TYPES and types & RIGHT_TYPES)

    def max_edge_length(self, cones: List[Cone]) -> float:
        p = [cones[i] for i in self.indices]
        d1 = np.hypot(p[0].x - p[1].x, p[0].y - p[1].y)
        d2 = np.hypot(p[1].x - p[2].x, p[1].y - p[2].y)
        d3 = np.hypot(p[2].x - p[0].x, p[2].y - p[0].y)
        return max(d1, d2, d3)


class Path:
    """
    Represents the FSA track path with cones, triangles, centerline, and racing line.

    Attributes:
        track_width (float): Nominal track width in meters.
        cones (List[Cone]): List of cones on the track.
        triangles (List[Triangle]): Filtered Delaunay triangles.
        centerline (List[Tuple[float, float]]): Raw centerline points.
        smoothed_centerline (List[Tuple[float, float]]): Smoothed centerline (Lap 1).
        racing_line (List[Tuple[float, float]]): Min-curvature racing line (Lap 2+).
    """

    def __init__(self, cones: List[Cone], track_width: float = 3.5):
        self.track_width = track_width
        self.cones = cones
        self.triangles: List[Triangle] = []
        self.centerline: List[Tuple[float, float]] = []
        self.smoothed_centerline: List[Tuple[float, float]] = []
        self.racing_line: List[Tuple[float, float]] = []

    def get_triangles(self) -> List[Triangle]:
        """Compute and return the filtered Delaunay triangles."""
        self._compute_triangles()
        return self.triangles

    def get_centerline(self) -> List[Tuple[float, float]]:
        """Compute and return the raw centerline waypoints."""
        self._compute_centerline()
        return self.centerline

    def get_smoothed_centerline(self, num_points: int = 200) -> List[Tuple[float, float]]:
        """Compute and return the smoothed centerline waypoints."""
        self._smooth_centerline(num_points)
        return self.smoothed_centerline

    def get_racing_line(self, safety_margin: float = 0.5) -> List[Tuple[float, float]]:
        """Compute and return the minimum-curvature racing line (shortest path)."""
        self._compute_racing_line(safety_margin)
        return self.racing_line

    def _compute_triangles(self):
        """Compute Delaunay triangulation and filter triangles."""
        pts = np.array([[c.x, c.y] for c in self.cones])
        simplices = Delaunay(pts, qhull_options='QJ').simplices
        self.triangles = [Triangle(tuple(t)) for t in simplices]
        self._filter_triangles()

    def _filter_triangles(self):
        """Filter triangles to keep only those connecting left and right boundaries."""
        max_edge = self.track_width * 2.5
        self.triangles = [
            t for t in self.triangles
            if t.has_mixed_types(self.cones) and t.max_edge_length(self.cones) <= max_edge
        ]

    def _compute_centerline(self):
        """Compute the centerline by finding cross-track edges and sorting them."""
        if not self.triangles:
            self._compute_triangles()
        cross_edges = self._find_cross_edges()
        if not cross_edges:
            self.centerline = []
            return
        midpts = self._compute_midpoints(cross_edges)
        adjacency = self._build_adjacency(cross_edges)
        sorted_indices = self._sort_centerline_points(midpts, adjacency)
        sorted_pts = midpts[sorted_indices]
        self.centerline = [(float(p[0]), float(p[1])) for p in sorted_pts]
        # Don't force close the loop for real-time paths

    def _find_cross_edges(self) -> Set[Tuple[int, int]]:
        """Find edges that cross between left and right boundaries."""
        cross_edges = set()
        for t in self.triangles:
            for i in range(3):
                a, b = t.indices[i], t.indices[(i + 1) % 3]
                ta, tb = self.cones[a].cone_type, self.cones[b].cone_type
                is_blue_yellow = (ta == "blue" and tb == "yellow") or (ta == "yellow" and tb == "blue")
                is_orange_cross = (
                    ta in {"small_orange", "large_orange"} and
                    tb in {"small_orange", "large_orange"} and
                    abs(self.cones[a].y - self.cones[b].y) > 0.01
                )
                if is_blue_yellow or is_orange_cross:
                    cross_edges.add((min(a, b), max(a, b)))
        return cross_edges

    def _compute_midpoints(self, cross_edges: Set[Tuple[int, int]]) -> np.ndarray:
        """Compute midpoints of cross edges."""
        return np.array([
            ((self.cones[e[0]].x + self.cones[e[1]].x) / 2,
             (self.cones[e[0]].y + self.cones[e[1]].y) / 2)
            for e in cross_edges
        ])

    def _build_adjacency(self, cross_edges: Set[Tuple[int, int]]) -> dict:
        """Build adjacency list for cross edges based on shared triangles."""
        edges_list = list(cross_edges)
        adjacency = {i: [] for i in range(len(edges_list))}
        for t in self.triangles:
            edges_in_t = []
            for i in range(3):
                a, b = t.indices[i], t.indices[(i + 1) % 3]
                edge = (min(a, b), max(a, b))
                if edge in cross_edges:
                    edges_in_t.append(edges_list.index(edge))
            for i in range(len(edges_in_t)):
                for j in range(i + 1, len(edges_in_t)):
                    idx_i, idx_j = edges_in_t[i], edges_in_t[j]
                    if idx_j not in adjacency[idx_i]:
                        adjacency[idx_i].append(idx_j)
                    if idx_i not in adjacency[idx_j]:
                        adjacency[idx_j].append(idx_i)
        return adjacency

    def _sort_centerline_points(self, midpts: np.ndarray, adjacency: dict) -> List[int]:
        """Sort centerline points using nearest neighbor starting from start position."""
        start_pos = np.array([self.cones[0].x, self.cones[0].y])
        start_idx = int(np.argmin(np.linalg.norm(midpts - start_pos, axis=1)))
        sorted_indices = [start_idx]
        remaining = set(range(len(midpts)))
        remaining.remove(start_idx)

        while remaining:
            curr = sorted_indices[-1]
            candidates = [n for n in adjacency[curr] if n in remaining]
            if candidates:
                dists = np.linalg.norm(midpts[candidates] - midpts[curr], axis=1)
                next_idx = candidates[np.argmin(dists)]
            else:
                rem_list = list(remaining)
                dists = np.linalg.norm(midpts[rem_list] - midpts[curr], axis=1)
                next_idx = rem_list[np.argmin(dists)]
            sorted_indices.append(next_idx)
            remaining.remove(next_idx)
        return sorted_indices

    def _smooth_centerline(self, num_points: int = 200):
        """Smooth the centerline using cubic spline interpolation."""
        if not self.centerline:
            self._compute_centerline()
        if len(self.centerline) < 4:
            self.smoothed_centerline = self.centerline
            return

        centerline_np = np.array(self.centerline)
        dx = np.diff(centerline_np[:, 0])
        dy = np.diff(centerline_np[:, 1])
        dist = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(dist)])
        mask = np.hstack([[True], dist > 1e-6])
        s, x, y = s[mask], centerline_np[:, 0][mask], centerline_np[:, 1][mask]

        if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            s = np.append(s, s[-1] + np.hypot(x[-2] - x[-1], y[-2] - y[-1]))

        cs_x = CubicSpline(s, x, bc_type='periodic')
        cs_y = CubicSpline(s, y, bc_type='periodic')
        s_new = np.linspace(0, s[-1], num_points)
        x_new = cs_x(s_new)
        y_new = cs_y(s_new)
        self.smoothed_centerline = list(zip(x_new, y_new))

    def _compute_racing_line(self, safety_margin: float = 0.5):
        """Compute the minimum-curvature racing line using quadratic programming."""
        if not self.smoothed_centerline:
            self._smooth_centerline(500)  # Use more points for racing line
        if len(self.smoothed_centerline) < 4:
            self.racing_line = self.smoothed_centerline
            return

        print("Computing minimum-curvature racing line …")
        pts = np.array(self.smoothed_centerline)
        n = len(pts)

        normals = self._compute_normals(pts)
        alpha_max, alpha_min = self._compute_boundary_offsets(pts, normals, safety_margin)
        Q, c_vec = self._build_qp_matrices(pts, normals)

        result = minimize(
            lambda alpha: float(alpha @ Q @ alpha + 2.0 * c_vec @ alpha),
            np.zeros(n),
            jac=lambda alpha: 2.0 * (Q @ alpha + c_vec),
            method="L-BFGS-B",
            bounds=list(zip(alpha_min, alpha_max)),
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not result.success:
            print(f"[WARNING] Optimiser: {result.message}")
        else:
            print(f"[OK] Optimiser converged in {result.nit} iterations.")

        alpha_opt = result.x
        racing_pts = pts + alpha_opt[:, np.newaxis] * normals
        self.racing_line = [(float(x), float(y)) for x, y in racing_pts]

        mean_offset = np.mean(np.abs(alpha_opt))
        max_offset = np.max(np.abs(alpha_opt))
        print(f"    Mean lateral offset: {mean_offset:.3f} m  |  Max: {max_offset:.3f} m")

    def _compute_normals(self, pts: np.ndarray) -> np.ndarray:
        """Compute unit outward normals at each point."""
        n = len(pts)
        tangents = np.array([pts[(i + 1) % n] - pts[(i - 1) % n] for i in range(n)])
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        tangents /= norms
        return np.column_stack([-tangents[:, 1], tangents[:, 0]])  # 90° CCW

    def _compute_boundary_offsets(self, pts: np.ndarray, normals: np.ndarray, safety_margin: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute lateral bounds for each point based on cone positions."""
        n = len(pts)
        default_hw = self.track_width / 2.0 - safety_margin
        alpha_max = np.full(n, default_hw)
        alpha_min = np.full(n, -default_hw)

        blue_pts = np.array([[c.x, c.y] for c in self.cones if c.cone_type == "blue"])
        yell_pts = np.array([[c.x, c.y] for c in self.cones if c.cone_type == "yellow"])

        if blue_pts.size == 0 or yell_pts.size == 0:
            return alpha_max, alpha_min

        for i in range(n):
            ni = normals[i]
            proj_blue = (blue_pts - pts[i]) @ ni
            pos_proj = proj_blue[proj_blue > 0]
            if pos_proj.size:
                alpha_max[i] = max(np.min(pos_proj) - safety_margin, 0.05)

            proj_yell = (yell_pts - pts[i]) @ ni
            neg_proj = proj_yell[proj_yell < 0]
            if neg_proj.size:
                alpha_min[i] = min(np.max(neg_proj) + safety_margin, -0.05)

        return alpha_max, alpha_min

    def _build_qp_matrices(self, pts: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build quadratic programming matrices for minimum curvature."""
        n = len(pts)
        nx, ny = normals[:, 0], normals[:, 1]
        xc, yc = pts[:, 0], pts[:, 1]

        B = np.zeros((2 * n, n))
        d2c = np.zeros(2 * n)

        for i in range(n):
            im1, ip1 = (i - 1) % n, (i + 1) % n
            B[2*i, im1] += nx[im1]
            B[2*i, i] += -2.0 * nx[i]
            B[2*i, ip1] += nx[ip1]
            d2c[2*i] = xc[ip1] - 2.0*xc[i] + xc[im1]

            B[2*i+1, im1] += ny[im1]
            B[2*i+1, i] += -2.0 * ny[i]
            B[2*i+1, ip1] += ny[ip1]
            d2c[2*i+1] = yc[ip1] - 2.0*yc[i] + yc[im1]

        Q = B.T @ B
        c = B.T @ d2c
        return Q, c

    # -----------------------------------------------------------------------
    # Plotting and Pipeline
    # -----------------------------------------------------------------------

    def plot(self, plot_triangles=False, plot_centerline=False, plot_smooth=True, plot_racing=True):
        """Plot the track, cones, smoothed centerline, and racing line."""
        color_map = {"blue": "blue", "yellow": "gold", "small_orange": "orange", "large_orange": "darkorange"}
        size_map = {"blue": 25, "yellow": 25, "small_orange": 80, "large_orange": 150}

        fig, ax = plt.subplots(figsize=(14, 9))
        pts = np.array([[c.x, c.y] for c in self.cones])

        if plot_triangles and self.triangles:
            for t in self.triangles:
                tp = pts[list(t.indices)]
                ax.fill(tp[:, 0], tp[:, 1], color="lightgrey", alpha=0.25, zorder=1)
                ax.plot(np.append(tp[:, 0], tp[0, 0]), np.append(tp[:, 1], tp[0, 1]), color="grey", lw=0.5, zorder=2)

        for c in self.cones:
            ax.scatter(c.x, c.y, color=color_map.get(c.cone_type, "black"), s=size_map.get(c.cone_type, 25),
                       edgecolors="black", linewidths=0.3, zorder=5)

        if plot_centerline and self.centerline:
            cx, cy = zip(*self.centerline)
            ax.plot(cx, cy, color="red", lw=1.5, zorder=6, label=f"Raw Centerline ({len(self.centerline)} pts)")

        if plot_smooth and self.smoothed_centerline:
            sx, sy = zip(*self.smoothed_centerline)
            ax.plot(sx, sy, color="limegreen", lw=2.0, zorder=7, label=f"Lap 1 — Smoothed Centerline ({len(self.smoothed_centerline)} pts)")

        if plot_racing and self.racing_line:
            rx, ry = zip(*self.racing_line)
            ax.plot(rx, ry, color="orangered", lw=2.5, zorder=8, label=f"Lap 2+ — Min-Curvature Racing Line ({len(self.racing_line)} pts)")

        ax.set_aspect("equal")
        ax.set_title("FS Autocross — Centerline (Lap 1) vs Racing Line (Lap 2+)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("track_output.png", dpi=150, bbox_inches="tight")
        print("Plot saved → track_output.png")
        plt.show()

    def run(self):
        """Full pipeline: triangulate → filter → centerline → smooth → racing line → plot."""
        self._compute_triangles()
        self._compute_centerline()
        self._smooth_centerline(500)  # Use more points for better racing line resolution
        self._compute_racing_line(0.5)
        self.plot()

        print(f"\nSummary:")
        print(f"  Cones      : {len(self.cones)}")
        print(f"  Triangles  : {len(self.triangles)}")
        print(f"  Centerline : {len(self.centerline)} pts (raw)  |  {len(self.smoothed_centerline)} pts (smoothed)")
        print(f"  Racing line: {len(self.racing_line)} pts")


# -----------------------------------------------------------------------
# Module Functions for Easy Import
# -----------------------------------------------------------------------

def get_cones(track_width: float = 3.5) -> List[Cone]:
    """Get the list of cones for the track."""
    return build_cones(track_width)


def get_triangles(track_width: float = 3.5) -> List[Triangle]:
    """Get the filtered Delaunay triangles."""
    cones = build_cones(track_width)
    path = Path(cones, track_width)
    return path.get_triangles()


def get_centerline(track_width: float = 3.5) -> List[Tuple[float, float]]:
    """Get the raw centerline waypoints."""
    cones = build_cones(track_width)
    path = Path(cones, track_width)
    return path.get_centerline()


def get_smoothed_centerline(track_width: float = 3.5, num_points: int = 200) -> List[Tuple[float, float]]:
    """Get the smoothed centerline waypoints."""
    cones = build_cones(track_width)
    path = Path(cones, track_width)
    return path.get_smoothed_centerline(num_points)


def get_racing_line(track_width: float = 3.5, safety_margin: float = 0.5) -> List[Tuple[float, float]]:
    """Get the minimum-curvature racing line (shortest path)."""
    cones = build_cones(track_width)
    path = Path(cones, track_width)
    return path.get_racing_line(safety_margin)


if __name__ == "__main__":
    cones = build_cones()
    path = Path(cones)
    path.run()
