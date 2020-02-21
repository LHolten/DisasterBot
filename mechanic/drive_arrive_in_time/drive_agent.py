from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive_arrive_in_time import DriveArriveInTime

from action.hit_ground_ball import HitGroundBall


class TestAgent(BaseTestAgent):
    def create_mechanic(self):
        return DriveArriveInTime(self, rendering_enabled=True)

    def get_mechanic_controls(self):

        target_loc, target_dt = HitGroundBall.get_target_ball_state(self.game_data)

        return self.mechanic.step(self.game_data.my_car, target_loc, target_dt)
