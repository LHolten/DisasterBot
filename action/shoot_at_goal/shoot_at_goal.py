from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from mechanic.drive_navigate_boost import DriveNavigateBoost
from util.ball_utils import get_ground_ball_intercept_state
from util.linear_algebra import normalize, norm


class ShootAtGoal(BaseAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveNavigateBoost(self.agent, self.rendering_enabled)

    def get_controls(self, game_data) -> SimpleControllerState:
        # logic for tepid hits
        target_dir = normalize(game_data.opp_goal.location - game_data.ball.location)
        target_dir[2] = 0
        box_location = game_data.ball.location - norm(game_data.ball.location - game_data.my_car.location) * target_dir
        box_location[2] = game_data.my_car.location[2]

        target_loc, target_dt = get_ground_ball_intercept_state(game_data, box_location)

        target_dir = normalize(game_data.opp_goal.location - target_loc)
        target_dir[2] = 0

        controls = self.mechanic.get_controls(
            game_data.my_car, game_data.boost_pads, target_loc, target_dt, target_dir.astype(float),
        )

        self.finished = self.mechanic.finished
        self.failed = self.mechanic.failed

        return controls

    def is_valid(self, game_data):
        return True
