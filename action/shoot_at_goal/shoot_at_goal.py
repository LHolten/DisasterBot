from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from mechanic.drive_navigate_boost import DriveNavigateBoost
from util.ball_utils import get_target_ball_state
from util.linear_algebra import dot, normalize


class ShootAtGoal(BaseAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveNavigateBoost(self.agent, rendering_enabled=self.rendering_enabled)

    def get_controls(self, game_data) -> SimpleControllerState:
        target_loc, target_dt = get_target_ball_state(game_data)

        target_vel = game_data.opp_goal.location - target_loc
        target_vel[2] = 0

        controls = self.mechanic.step(
            game_data.my_car, game_data.boost_pads, target_loc, target_dt, target_vel.astype(float),
        )

        self.finished = self.mechanic.finished
        self.failed = self.mechanic.failed

        return controls

    def is_valid(self, game_data):
        return True
