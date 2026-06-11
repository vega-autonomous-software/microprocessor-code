import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class Cone:
    """Represents a cone with position and type."""
    x: float
    y: float
    cone_type: str  # "blue" | "yellow" | "small_orange" | "large_orange"


def build_cones(track_width: float = 3.5) -> List[Cone]:
    """
    Build the FSA track cones: rectangle with rounded hairpin corners.

    Args:
        track_width: Width of the track in meters.

    Returns:
        List of Cone objects representing the track layout.
    """
    hw = track_width / 2
    R = 7.0  # Radius of hairpin turns
    SL = 50.0  # Length of straight sections
    sp = 3.0  # Spacing between cones
    cones = []

    # Bottom straight section
    for x in np.arange(sp, SL, sp):
        cones.append(Cone(x, hw, "blue"))
        cones.append(Cone(x, -hw, "yellow"))

    # Right hairpin (CW turn)
    cx_r, cy_r = SL, R
    for ang in np.linspace(np.radians(-90), np.radians(90), 37):
        ox, oy = np.cos(ang), np.sin(ang)
        px, py = cx_r + R * ox, cy_r + R * oy
        cones.append(Cone(px - ox * hw, py - oy * hw, "blue"))
        cones.append(Cone(px + ox * hw, py + oy * hw, "yellow"))

    # Top straight section
    for x in np.arange(SL - sp, 0, -sp):
        cones.append(Cone(x, 2 * R - hw, "blue"))
        cones.append(Cone(x, 2 * R + hw, "yellow"))

    # Left hairpin (CCW turn)
    cx_l, cy_l = 0.0, R
    for ang in np.linspace(np.radians(90), np.radians(270), 37):
        ox, oy = np.cos(ang), np.sin(ang)
        px, py = cx_l + R * ox, cy_l + R * oy
        cones.append(Cone(px - ox * hw, py - oy * hw, "blue"))
        cones.append(Cone(px + ox * hw, py + oy * hw, "yellow"))

    # Set start/finish markers
    for i in range(4):
        cones[i].cone_type = "small_orange"
    cones[-2].cone_type = "large_orange"
    cones[-1].cone_type = "large_orange"

    return cones
