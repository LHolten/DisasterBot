from mechanic.base_test_agent import BaseTestAgent
from mechanic.jumping_shot import JumpingShot


class TestAgent(BaseTestAgent):
    def create_mechanic(self):
        return JumpingShot(
            self,
            self.game_data.ball.location,
            self.game_data.time + (self.game_data.ball.location[2] * 0.333 * 0.00833),
            self.game_data.game_tick_packet,
            rendering_enabled=False,
        )

    def get_mechanic_controls(self):
        return self.mechanic.get_controls(self.game_data.my_car, self.game_data.game_tick_packet)
