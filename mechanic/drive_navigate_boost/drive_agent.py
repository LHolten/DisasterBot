from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive_navigate_boost.drive_navigate_boost import DriveNavigateBoost


class TestAgent(BaseTestAgent):
    def create_mechanic(self):
        return DriveNavigateBoost(self)

    def get_mechanic_controls(self):
        target_loc = self.game_data.ball.location
        car = self.game_data.my_car

        return self.mechanic.step(car, self.game_data.boost_pads, target_loc)
