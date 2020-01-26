from mechanic.base_test_agent import BaseTestAgent
from mechanic.spikerush_flick.flick import Flick
from rlbot.utils.game_state_util import GameState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class TestAgent(BaseTestAgent):
    game_state = None

    def create_mechanic(self):
        return Flick(self.info)

    def test_process(self, game_tick_packet: GameTickPacket):
        return True

    def get_mechanic_controls(self):
        target = self.info.ball.location - self.info.my_car.location
        return self.mechanic.step(target, self.info.time_delta)
