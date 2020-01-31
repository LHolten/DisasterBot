import math
import numpy as np


def norm(vec: np.ndarray):
    return math.sqrt(math.pow(vec[0], 2) + math.pow(vec[1], 2) + math.pow(vec[2], 2))


def normalize(vec: np.ndarray):
    """Returns a normalized vector of norm 1."""
    return vec / max(norm(vec), 1e-9)
