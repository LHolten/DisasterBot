from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive_navigate_boost.drive_navigate_boost import DriveNavigateBoost
from action.hit_ground_ball import HitGroundBall


class TestAgent(BaseTestAgent):
    def create_mechanic(self):
        return DriveNavigateBoost(self)

    def get_mechanic_controls(self):

        target_loc, target_dt = HitGroundBall.get_target_ball_state(self.game_data)

        return self.mechanic.step(self.game_data.my_car, self.game_data.boost_pads, target_loc)
