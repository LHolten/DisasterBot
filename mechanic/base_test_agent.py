from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlutilities.simulation import Game
from .base_mechanic import BaseMechanic


class BaseTestAgent(BaseAgent):

    def __init__(self, name, team, index):
        super(BaseTestAgent, self).__init__(name, team, index)
        self.info = Game(index, team)
        self.mechanic = self.create_mechanic()

    def get_output(self, game_tick_packet: GameTickPacket) -> SimpleControllerState:
        self.info.read_game_information(game_tick_packet,
                                        self.get_rigid_body_tick(),
                                        self.get_field_info())
        self.test_process(game_tick_packet)
        return self.get_mechanic_controls()

    def create_mechanic(self) -> BaseMechanic:
        raise NotImplementedError

    def get_mechanic_controls(self) -> SimpleControllerState:
        raise NotImplementedError

    def test_process(self, game_tick_packet: GameTickPacket):
        pass

    def initialize_agent(self):
        pass
