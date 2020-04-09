from mechanic.base_test_agent import BaseTestAgent
from mechanic.flip import Flip


class TestAgent(BaseTestAgent):
    def create_mechanic(self):
        return Flip(self, rendering_enabled=True)

    def get_mechanic_controls(self):
        target_dir = self.game_data.ball.location - self.game_data.my_car.location

        if self.mechanic.finished:
            self.mechanic = self.create_mechanic()

        return self.mechanic.get_controls(self.game_data.my_car, target_dir)
