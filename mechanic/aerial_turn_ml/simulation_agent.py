from rlbot.utils.game_state_util import GameState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from mechanic.base_test_agent import BaseTestAgent


class TestAgent(BaseTestAgent):
    game_state = None

    def create_mechanic(self):
        from mechanic.aerial_turn_ml.aerial_turn_ml import AerialTurnML
        return AerialTurnML()

    def test_process(self, game_tick_packet: GameTickPacket):
        run = False

        if not self.game_state and not self.matchcomms.incoming_broadcast.empty():
            self.matchcomms.incoming_broadcast.get_nowait()
            self.game_state = GameState.create_from_gametickpacket(game_tick_packet)
            self.game_state.cars[self.index].jumped = None
            self.game_state.cars[self.index].double_jumped = None
            self.game_state.cars[self.index].physics.rotation = None
            self.game_state.cars[self.index].physics.angular_velocity = None
            self.game_state.cars[self.index].physics.velocity.z = 10
            self.game_state.ball.physics.velocity.z = 10

        if self.game_state:
            self.set_game_state(self.game_state)
            run = True

        if self.game_state and self.mechanic.finished:
            self.matchcomms.outgoing_broadcast.put_nowait('done')
            self.game_state = None

        return run

    def get_mechanic_controls(self):
        return self.mechanic.step(self.info)
