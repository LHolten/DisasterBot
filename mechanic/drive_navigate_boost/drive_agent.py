from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive_navigate_boost.drive_navigate_boost import DriveNavigateBoost

import numpy as np

from skeleton.util.structure.dtypes import dtype_full_boost
from util.ball_utils import get_ground_ball_intercept_state


class TestAgent(BaseTestAgent):
    def create_mechanic(self):
        return DriveNavigateBoost(self, rendering_enabled=True)

    def get_mechanic_controls(self):
        target_loc, target_dt = get_ground_ball_intercept_state(self.game_data)

        own_goal = np.zeros(1, dtype_full_boost)
        own_goal["location"] = self.game_data.own_goal.location
        own_goal["timer"] = -np.inf

        nodes = np.concatenate([self.game_data.boost_pads, own_goal])

        return self.mechanic.step(
            self.game_data.my_car, nodes, target_loc, target_dt, self.game_data.opp_goal.location.astype(np.float64)
        )
