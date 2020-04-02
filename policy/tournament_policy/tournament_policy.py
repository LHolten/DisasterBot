import math

import numpy as np

from action.base_action import BaseAction
from action.collect_boost import CollectBoost
from action.hit_ground_ball import HitGroundBall
from action.kickoff import Kickoff
from action.shadow_ball import ShadowBall
from action.shoot_at_goal import ShootAtGoal
from policy.base_policy import BasePolicy
from skeleton.util.structure import GameData
from util.physics.drive_1d_heuristic import state_at_distance_heuristic


def get_ball_control(game_data):
    own_time, _, _ = state_at_distance_heuristic(
        game_data.my_car.location - game_data.ball.location, game_data.my_car.velocity, game_data.my_car.boost
    )
    teammate_time = np.inf
    if len(game_data.teammates) > 1:
        teammate_time = min(
            state_at_distance_heuristic(
                teammate["physics"]["location"].astype(float) - game_data.ball.location,
                teammate["physics"]["velocity"].astype(float),
                teammate["boost"].astype(float),
            )[0]
            for teammate in game_data.teammates
            if teammate["spawn_id"] != game_data.my_car.spawn_id
        )

    opponent_time = np.inf
    if len(game_data.opponents) > 0:
        opponent_time = min(
            state_at_distance_heuristic(
                opponent["physics"]["location"].astype(float) - game_data.ball.location,
                opponent["physics"]["velocity"].astype(float),
                opponent["boost"].astype(float),
            )[0]
            for opponent in game_data.opponents
        )

    if own_time < teammate_time and own_time < opponent_time:
        return "self"
    if opponent_time < teammate_time:
        return "opponent"
    return "teammate"


class TournamentPolicy(BasePolicy):
    def __init__(self, agent, rendering_enabled=True):
        super(TournamentPolicy, self).__init__(agent, rendering_enabled)
        self.kickoff_action = Kickoff(agent, rendering_enabled)
        self.attack = ShootAtGoal(agent, rendering_enabled)
        self.hit_ball = HitGroundBall(agent, rendering_enabled)
        self.shadow = ShadowBall(agent, rendering_enabled)
        self.collect_boost = CollectBoost(agent, rendering_enabled)

    def get_action(self, game_data: GameData) -> BaseAction:
        ball_loc = game_data.ball.location
        kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 1

        if kickoff:
            return self.kickoff_action
        else:
            control = get_ball_control(game_data)
            if control == "opponent":
                return self.defend(game_data)
            elif control == "teammate":
                return self.collect_boost
            else:
                return self.attack

    def defend(self, game_data):
        if np.linalg.norm(game_data.own_goal.location - game_data.ball.location) < 3000:
            return self.hit_ball
        return self.shadow
