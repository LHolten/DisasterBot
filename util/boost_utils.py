from rlutilities.simulation import Pad
from rlutilities.linear_algebra import vec3, norm


def closest_available_boost(my_pos: vec3, boost_pads: list) -> Pad:
    """ Returns the closest available boost pad to my_pos"""
    distance = [norm(pad.location - my_pos) for pad in boost_pads]
    min_index = distance.index(min(distance))

    return boost_pads[min_index]
