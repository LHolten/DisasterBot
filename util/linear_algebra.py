import math
import numpy as np


def norm(vec: np.ndarray):
    return math.sqrt(math.pow(vec[0], 2) + math.pow(vec[1], 2) + math.pow(vec[2], 2))


def dot(vec: np.ndarray, vec2: np.ndarray):
    return vec[0] * vec2[0] + vec[1] * vec2[1] + vec[2] * vec2[2]


def normalize(vec: np.ndarray):
    """Returns a normalized vector of norm 1."""
    return vec / max(norm(vec), 1e-8)


def normalize_batch(vec: np.ndarray):
    return vec / np.maximum(np.linalg.norm(vec, axis=-1), 1e-8)


def angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray):
    """Provides angle between 2 vectors in radians"""
    return dot(normalize(vec1), normalize(vec2)) * math.pi


def optimal_intercept_vector(collider_location: np.ndarray, collider_velocity: np.ndarray, target_location: np.ndarray):
    """Provides vector for correcting an object's velocity vector towards the target vector"""
    return normalize(normalize(collider_velocity) + normalize(collider_location - target_location))
