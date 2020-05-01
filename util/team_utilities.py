import numpy as np
from skeleton.util.structure import GameData
from util.linear_algebra import norm, normalize, optimal_intercept_vector


def kickoff_decider(game_data: GameData) -> bool:
    if len(game_data.teammates) > 1:
        my_distance = np.linalg.norm(game_data.my_car.location - game_data.ball.location)
        for ally in game_data.teammates:
            if np.linalg.norm(ally[0][0] - game_data.ball.location) < my_distance:
                return False
    return True


def calc_target_dir(game_data: GameData, ball_location, ball_velocity):
    own_goal = game_data.own_goal.location
    opp_goal = game_data.opp_goal.location

    relative_own = ball_location - own_goal
    relative_opp = opp_goal - ball_location

    opp_target_dir = optimal_intercept_vector(ball_location, ball_velocity, opp_goal,)

    return norm(relative_own) * normalize(opp_target_dir) + norm(relative_opp) * normalize(relative_own)
