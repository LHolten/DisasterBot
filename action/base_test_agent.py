from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlutilities.simulation import Game
from .base_action import BaseAction


class BaseTestAgent(BaseAgent):

    def __init__(self, name, team, index):
        super(BaseTestAgent, self).__init__(name, team, index)
        self.info = Game(index, team)
        self.action = self.create_action()

    def get_output(self, game_tick_packet: GameTickPacket) -> SimpleControllerState:
        self.info.read_game_information(game_tick_packet,
                                        self.get_rigid_body_tick(),
                                        self.get_field_info())
        self.action.update_status(self.info)
        self.test_process(game_tick_packet)
        return self.action.get_output(self.info)

    def create_action(self) -> BaseAction:
        raise NotImplementedError

    def test_process(self, game_tick_packet: GameTickPacket):
        pass

    def initialize_agent(self):
        pass
