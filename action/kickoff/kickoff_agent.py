from action.base_test_agent import BaseTestAgent
from action.kickoff.kickoff import Kickoff
from rlbot.utils.structures.game_data_struct import GameTickPacket


class TestAgent(BaseTestAgent):
    initialized = False

    def test_process(self, game_tick_packet: GameTickPacket):
        if not self.initialized and not self.matchcomms.incoming_broadcast.empty():
            self.matchcomms.incoming_broadcast.get_nowait()
            self.initialized = True

        if self.initialized:
            self.action.update_status(self.info)

        if self.initialized and self.action.finished:
            self.matchcomms.outgoing_broadcast.put_nowait('done')
            self.initialized = False

    def create_action(self):
        return Kickoff(self.renderer)
