from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from mechanic.drive_navigate_boost import DriveNavigateBoost
from mechanic.jumping_shot import JumpingShot
from util.ball_utils import get_ground_ball_intercept_state, get_high_ball_intercept_state
from util.linear_algebra import normalize, norm
import numpy as np


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

        air_target_loc, air_target_dt = get_high_ball_intercept_state(game_data, box_location)
        ground_target_loc, ground_target_dt = get_ground_ball_intercept_state(game_data, box_location)

        if air_target_dt < ground_target_dt:
            target_loc = air_target_loc
            target_dt = air_target_dt

            if self.jumpshot_valid(game_data, target_loc, target_dt):
                # activate jumpshot mechanic
                pass
        else:
            target_loc = ground_target_loc
            target_dt = ground_target_dt

        target_dir = normalize(game_data.opp_goal.location - target_loc)
        target_dir[2] = 0

        controls = self.mechanic.get_controls(
            game_data.my_car, game_data.boost_pads, target_loc, target_dt, target_dir.astype(float),
        )

        self.finished = self.mechanic.finished
        self.failed = self.mechanic.failed

        return controls

    def jumpshot_valid(self, game_data, target_loc, target_time) -> bool:
        hopefully_temp_distance_limit = 120
        temp_time_limit = 0.85
        future_projection = np.array([game_data.my_car.location.x, game_data.my_car.location.y, 0]) + np.array(
            [game_data.my_car.velocity.x, game_data.my_car.velocity.y, 0]
        ) * (target_time - game_data.game_info.seconds_elapsed)
        if np.linalg.norm(target_loc - future_projection - target_loc) < hopefully_temp_distance_limit:
            if target_time - game_data.game_info.seconds_elapsed < temp_time_limit:
                return True

        return False

    def is_valid(self, game_data):
        return True
