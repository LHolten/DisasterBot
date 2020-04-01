from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from mechanic.drive_navigate_boost import DriveNavigateBoost
from util.ball_utils import get_ground_ball_intercept_state
from util.linear_algebra import normalize


class ShootAtGoal(BaseAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveNavigateBoost(self.agent, rendering_enabled=self.rendering_enabled)

    def get_controls(self, game_data) -> SimpleControllerState:
        target_loc, target_dt = get_ground_ball_intercept_state(game_data)

        target_dir = normalize(game_data.opp_goal.location - target_loc)
        target_dir[2] = 0

        controls = self.mechanic.step(
            game_data.my_car, game_data.boost_pads, target_loc, target_dt, target_dir.astype(float),
        )

        self.finished = self.mechanic.finished
        self.failed = self.mechanic.failed

        return controls

    def is_valid(self, game_data):
        return True
