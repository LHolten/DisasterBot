from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive_arrive_in_time_with_vel import DriveArriveInTimeWithVel

from util.ball_utils import get_target_ball_state


class TestAgent(BaseTestAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_time = None

    def create_mechanic(self):
        return DriveArriveInTimeWithVel(self, rendering_enabled=True)

    def get_mechanic_controls(self):

        target_loc = get_target_ball_state(self.game_data)[0]

        if self.target_time is None:
            self.target_time = self.game_data.time + 6

        target_dt = self.target_time - self.game_data.time
        final_vel = 2300

        controls = self.mechanic.step(self.game_data.my_car, target_loc, target_dt, final_vel)

        if self.mechanic.finished or self.mechanic.failed:
            self.target_time = None

        return controls
