import numpy as np
from random import uniform

from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive_arrive_in_time_with_vel import DriveArriveInTimeWithVel


class TestAgent(BaseTestAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_loc = None
        self.target_time = None

    def create_mechanic(self):
        return DriveArriveInTimeWithVel(self, rendering_enabled=True)

    def get_mechanic_controls(self):

        if self.target_loc is None:
            # remove "or True" to test the accuracy without recalculating each tick.
            self.target_loc = np.array([uniform(-1, 1) * 1000, uniform(-1, 1) * 1000, 17])
            self.target_time = self.game_data.time + 6

        target_dt = self.target_time - self.game_data.time
        final_vel = 2300

        controls = self.mechanic.step(self.game_data.my_car, self.target_loc, target_dt, final_vel)

        if self.mechanic.finished or self.mechanic.failed:
            self.target_loc = None

        return controls
