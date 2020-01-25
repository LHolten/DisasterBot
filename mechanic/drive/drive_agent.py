import numpy as np

from skeleton.util.conversion import vector3_to_numpy

from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive import DriveTurnToFaceTarget


class TestAgent(BaseTestAgent):

    def create_mechanic(self):
        return DriveTurnToFaceTarget(self)

    def get_mechanic_controls(self):
        target_loc = self.game_data.ball.location
        return self.mechanic.step(self.game_data.my_car, target_loc)
