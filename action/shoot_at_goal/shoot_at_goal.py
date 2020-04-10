from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from mechanic.drive_navigate_boost import DriveNavigateBoost
from mechanic.jumping_shot import JumpingShot
from util.ball_utils import get_ground_ball_intercept_state
from util.linear_algebra import normalize, norm, optimal_intercept_vector
import numpy as np


class ShootAtGoal(BaseAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveNavigateBoost(self.agent, self.rendering_enabled)
        self.jump_shot = None

    def get_controls(self, game_data) -> SimpleControllerState:
        target_dir = optimal_intercept_vector(
            game_data.ball.location, game_data.ball.velocity, game_data.opp_goal.location
        )
        target_dir[2] = 0

        target_loc, target_dt = get_ground_ball_intercept_state(game_data)

        if target_loc[2] > 100 and self.jumpshot_valid(game_data, target_loc, target_dt) and self.jump_shot is None:
            self.jump_shot = JumpingShot(
                self.agent, target_loc, target_dt - 0.1, game_data.game_tick_packet, self.rendering_enabled,
            )

        if self.jump_shot is not None:
            controls = self.jump_shot.get_controls(game_data.my_car, game_data.game_tick_packet)
            self.finished = self.jump_shot.finished
            self.failed = self.jump_shot.failed
            return controls

        target_dir = normalize(game_data.opp_goal.location - target_loc)
        target_dir[2] = 0

        controls = self.mechanic.get_controls(
            game_data.my_car, game_data.boost_pads, target_loc, target_dt, target_dir.astype(float),
        )

        self.finished = self.mechanic.finished
        self.failed = self.mechanic.failed

        return controls

    def jumpshot_valid(self, game_data, target_loc, target_dt) -> bool:
        temp_distance_limit = 120
        temp_time_limit = target_loc[2] / 300
        future_projection = game_data.my_car.location + game_data.my_car.velocity * target_dt
        difference = future_projection - target_loc
        difference[2] = 0
        if np.linalg.norm(difference) < temp_distance_limit and target_dt < temp_time_limit:
            return True
        return False

    def is_valid(self, game_data):
        return True
