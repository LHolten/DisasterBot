import math
import numpy as np

PI = math.pi


class Spherical:

    """Simple class holding spehrical coordinates"""

    def __init__(self, radius: float = 1, inclination: float = 0.0, azimuth: float = 0.0):
        self.radius = radius
        self.inclination = inclination
        self.azimuth = azimuth


def range_180(angle: float, pi_unit: float = PI) -> float:
    """Wraps any angle to [-pi_unit, pi_unit] range, example: Range180(270, 180) = -90"""
    if abs(angle) >= 2 * pi_unit:
        angle -= abs(angle) // (2 * pi_unit) * 2 * pi_unit * math.copysign(1, angle)
    if abs(angle) > pi_unit:
        angle -= 2 * pi_unit * math.copysign(1, angle)
    return angle


def to_spherical(vec: np.ndarray) -> Spherical:
    """Converts from cartesian to spherical coordinates."""
    radius = np.linalg.norm(vec) + 1e-9
    inclination = math.acos(vec[2] / radius)
    azimuth = math.atan2(vec[1], vec[0])
    return Spherical(radius, range_180(PI / 2 - inclination), azimuth)
