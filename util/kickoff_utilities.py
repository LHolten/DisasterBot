import numpy as np
from skeleton.util.structure import GameData


def kickoff_decider(game_data: GameData) -> bool:
    if len(game_data.teammates) > 1:
        my_distance = np.linalg.norm(game_data.my_car.location - game_data.ball.location)
        for ally in game_data.teammates:
            if np.linalg.norm(ally[0][0] - game_data.ball.location) < my_distance:
                return False
    return True


def get_kickoff_position(position: np.array):
    # kickoff_locations = [[2048, 2560], [256, 3848], [0, 4608]]
    if abs(position[0]) >= 300:
        return 0  # wide diagonal
    elif abs(position[0]) > 5:
        return 1  # short diagonal
    else:
        return 2  # middle
