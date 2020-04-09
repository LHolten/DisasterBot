from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive_turn_face_target import DriveTurnFaceTarget


class TestAgent(BaseTestAgent):
    def create_mechanic(self):
        return DriveTurnFaceTarget(self, rendering_enabled=True)

    def get_mechanic_controls(self):

        return self.mechanic.get_controls(self.game_data.my_car, self.game_data.ball.location)
