import numpy as np

from util.linear_algebra import normalize_batch


HITBOX = np.array([59, 42, 18])
HITBOX_OFFSET = np.array([13.87566, 0, 20.755])
HITBOX.flags.writeable = False
HITBOX_OFFSET.flags.writeable = False


def box_local_collision_location(local_point: np.ndarray, box_corner=HITBOX, box_offset=HITBOX_OFFSET):
    """point of contact in local coordinates"""
    return np.clip(local_point - box_offset, -box_corner, box_corner) + box_offset


def box_point_collision_location(point_loc, box_loc, box_rot_matrix, box_corner=HITBOX, box_offset=HITBOX_OFFSET):
    """Point of hypothetical collision on the box."""
    point_local_loc = (point_loc - box_loc).dot(box_rot_matrix)
    collision_local_loc = box_local_collision_location(point_local_loc, box_corner, box_offset)
    return box_loc + box_rot_matrix.dot(collision_local_loc)


def box_ball_location_on_collision(
    ball_loc, box_loc, box_rot_matrix, box_corner=HITBOX, box_offset=HITBOX_OFFSET, ball_radius=92
):
    """Closest box location for there to be a collision with the ball"""
    ball_local_loc = (ball_loc - box_loc).dot(box_rot_matrix)
    hit_local_loc = box_local_collision_location(ball_local_loc, box_corner, box_offset)
    box_local_location_on_collision = (
        ball_local_loc + normalize_batch(hit_local_loc - ball_local_loc) * ball_radius - hit_local_loc
    )
    return box_rot_matrix.dot(box_local_location_on_collision) + box_loc


def box_ball_collision_distance(
    ball_loc, box_loc, box_rot_matrix, box_corner=HITBOX, box_offset=HITBOX_OFFSET, ball_radius=92
):
    """Distance left until collision"""
    ball_local_loc = (ball_loc - box_loc).dot(box_rot_matrix)
    hit_local_loc = box_local_collision_location(ball_local_loc, box_corner, box_offset)
    return np.linalg.norm(hit_local_loc - ball_local_loc, axis=-1) - ball_radius
