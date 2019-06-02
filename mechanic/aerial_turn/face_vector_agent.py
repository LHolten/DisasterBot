from mechanic.base_test_agent import BaseTestAgent
from mechanic.aerial_turn.face_vector import FaceVectorRLU
from rlbot.utils.game_state_util import GameState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class TestAgent(BaseTestAgent):
    game_state = None

    def create_mechanic(self):
        return FaceVectorRLU()

    def test_process(self, game_tick_packet: GameTickPacket):
        if not self.game_state and not self.matchcomms.incoming_broadcast.empty():
            self.matchcomms.incoming_broadcast.get_nowait()
            self.game_state = GameState.create_from_gametickpacket(game_tick_packet)
            self.game_state.cars[self.index].physics.rotation = None
            self.game_state.cars[self.index].physics.angular_velocity = None
            self.game_state.cars[self.index].physics.velocity.z = 10
            self.game_state.ball.physics.velocity.z = 10

        if self.game_state:
            self.set_game_state(self.game_state)

        if self.game_state and self.mechanic.finished:
            self.matchcomms.outgoing_broadcast.put_nowait('done')
            self.game_state = None

    def get_mechanic_controls(self):
        target = self.info.ball.location - self.info.my_car.location
        return self.mechanic.step(self.info.my_car, target, self.info.time_delta)