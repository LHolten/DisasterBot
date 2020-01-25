from rlbot.utils.structures.game_data_struct import Vector3, Rotator
import numpy as np
import math


def vector3_to_numpy(vector3: Vector3):
    """Converts Vector3 to numpy array"""
    a = np.array([vector3.x, vector3.y, vector3.z])
    a.flags.writeable = False  # this makes the numpy array immutable
    return a


def rotator_to_numpy(rotator: Rotator):
    """Converts rotator to numpy array"""
    a = np.array([rotator.pitch, rotator.yaw, rotator.roll])
    a.flags.writeable = False
    return a


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
