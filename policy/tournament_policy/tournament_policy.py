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
from util.linear_algebra import norm
from util.numerics import sign
from util.physics.drive_1d_heuristic import state_at_distance_heuristic
from util.team_utilities import kickoff_decider


def get_ball_control(game_data: GameData):
    own_time, _, _ = state_at_distance_heuristic(
        game_data.my_car.location - game_data.ball.location, game_data.my_car.velocity, game_data.my_car.boost
    )
    team_sign = sign(game_data.team)

    def is_active_team(teammate):
        teammate_location = teammate["physics"]["location"].astype(float)
        return (game_data.ball.location[1] - teammate_location[1]) * team_sign < 0

    def time_to_ball(player):
        return state_at_distance_heuristic(
            player["physics"]["location"].astype(float) - game_data.ball.location,
            player["physics"]["velocity"].astype(float),
            player["boost"].astype(float),
        )[0]

    teammates = list(filter(is_active_team, game_data.teammates))

    teammate_time = min([*map(time_to_ball, teammates), np.inf])
    teammate_max = max([*map(time_to_ball, teammates), 0])
    opponent_time = min([*map(time_to_ball, game_data.opponents), np.inf])

    return own_time, teammate_time, opponent_time, len(teammates), teammate_max


class TournamentPolicy(BasePolicy):
    def __init__(self, agent, rendering_enabled=True):
        super(TournamentPolicy, self).__init__(agent, rendering_enabled)
        self.kickoff_action = Kickoff(agent, rendering_enabled)
        self.attack = ShootAtGoal(agent, rendering_enabled)
        self.hit_ball = HitGroundBall(agent, rendering_enabled)
        self.shadow = ShadowBall(agent, rendering_enabled, 1500)
        self.shadow2 = ShadowBall(agent, rendering_enabled, 3000)
        self.collect_boost = CollectBoost(agent, rendering_enabled)

    def get_action(self, game_data: GameData) -> BaseAction:
        ball_loc = game_data.ball.location
        kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 1

        if kickoff:
            if kickoff_decider(game_data):
                return self.kickoff_action
            else:
                return self.shadow
        else:
            own, team, opp, num_active, team_max = get_ball_control(game_data)
            if own == team and (
                num_active > 1 or own < opp or norm(game_data.own_goal.location - game_data.ball.location) < 3000
            ):
                if self.attack.finished:
                    self.attack = ShootAtGoal(self.agent, self.rendering_enabled)
                return self.attack
            else:
                if num_active > 2 and own == team_max:
                    return self.shadow2
                return self.shadow
