import math
import numpy as np

from rlbot.utils.structures.game_data_struct import Vector3, Rotator, BoxShape


def vector3_to_numpy(vector: Vector3):
    """Converts Vector3 to numpy array"""
    return np.array([vector.x, vector.y, vector.z])


def rotator_to_numpy(rotator: Rotator):
    """Converts rotator to numpy array"""
    return np.array([rotator.pitch, rotator.yaw, rotator.roll])


def rotator_to_matrix(rotator: Rotator):
    """Converts Rotator to numpy matrix"""
    CP = math.cos(rotator.pitch)
    SP = math.sin(rotator.pitch)
    CY = math.cos(rotator.yaw)
    SY = math.sin(rotator.yaw)
    CR = math.cos(rotator.roll)
    SR = math.sin(rotator.roll)

    theta = np.zeros((3, 3))

    # front direction
    theta[0, 0] = CP * CY
    theta[1, 0] = CP * SY
    theta[2, 0] = SP

    # left direction
    theta[0, 1] = CY * SP * SR - CR * SY
    theta[1, 1] = SY * SP * SR + CR * CY
    theta[2, 1] = -CP * SR

    # up direction
    theta[0, 2] = -CR * CY * SP - SR * SY
    theta[1, 2] = -CR * SY * SP + SR * CY
    theta[2, 2] = CP * CR

    theta.flags.writeable = False

    return theta


def box_shape_to_numpy(box_shape: BoxShape):
    """Converts BoxShape to numpy array"""
    return np.array([box_shape.length, box_shape.width, box_shape.height])
