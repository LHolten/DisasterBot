from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from mechanic.drive_arrive_in_time_with_vel import DriveArriveInTimeWithVel
from skeleton.util.structure import GameData
from util.linear_algebra import normalize, norm, dot


class ShadowBall(BaseAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveArriveInTimeWithVel(self.agent, rendering_enabled=self.rendering_enabled)

    def get_controls(self, game_data: GameData) -> SimpleControllerState:
        target_loc = normalize(game_data.own_goal.location - game_data.ball.location) * 1200 + game_data.ball.location
        up = game_data.my_car.rotation_matrix[:, 2]
        target_loc = target_loc - dot(target_loc - game_data.my_car.location, up) * up
        target_vel = norm(game_data.ball.velocity - dot(game_data.ball.velocity, up) * up)

        controls = self.mechanic.step(game_data.my_car, target_loc, 0.5, target_vel)

        self.finished = self.mechanic.finished
        self.failed = self.mechanic.failed

        return controls

    def is_valid(self, game_data):
        return True
