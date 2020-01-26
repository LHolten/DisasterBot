from typing import List
import numpy as np

from skeleton.util.structure.game_data import Pad


def closest_available_boost(my_loc: np.ndarray, boost_pads: List[Pad]) -> Pad:
    """Returns the closest available boost pad to my_loc"""
    closest = None
    for boost in boost_pads:
        distance = np.linalg.norm(boost.location - my_loc)
        boost_time_to_recharge = 10 if boost.is_large else 4
        if boost.is_active or distance / 2300 > boost_time_to_recharge - boost.timer:
            if closest is None or distance < closest['distance']:
                closest = {'boost': boost, 'distance': distance}

    return closest['boost']
