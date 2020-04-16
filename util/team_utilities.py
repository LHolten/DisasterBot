import numpy as np
from skeleton.util.structure import GameData


def kickoff_decider(game_data: GameData) -> bool:
    if len(game_data.teammates) > 1:
        my_distance = np.linalg.norm(game_data.my_car.location - game_data.ball.location)
        for ally in game_data.teammates:
            if np.linalg.norm(ally[0][0] - game_data.ball.location) < my_distance:
                return False
    return True
