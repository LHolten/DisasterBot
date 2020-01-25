from typing import List
import numpy as np

from skeleton.util.structure.game_data import Pad


def closest_available_boost(my_loc: np.ndarray, boost_pads: List[Pad]) -> Pad:
    """Returns the closest available boost pad to my_loc"""
    closest_boost = None
    for boost in boost_pads:
        distance = np.linalg.norm(boost.location - my_loc)
        boost_time_to_recharge = 10 if boost.is_large else 4
        if boost.is_active or distance / 2300 > boost_time_to_recharge - boost.timer:
            if closest_boost is None:
                closest_boost = boost
                closest_distance = np.linalg.norm(closest_boost.location - my_loc)
            else:
                if distance < closest_distance:
                    closest_boost = boost
                    closest_distance = np.linalg.norm(closest_boost.location - my_loc)
    return closest_boost
