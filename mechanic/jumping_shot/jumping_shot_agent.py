import numpy as np
import math
from mechanic.base_test_agent import BaseTestAgent
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator
from util.linear_algebra import normalize
from mechanic.jumping_shot import JumpingShot


class TestAgent(BaseTestAgent):
    def create_mechanic(self):
        #def __init__(self, agent, target_loc, target_time, game_packet, rendering_enabled=False)
        return JumpingShot(
            self,
            self.game_data.ball.location,
            self.game_data.game_tick_packet.game_info.seconds_elapsed + (self.game_data.ball.location[2] * 0.333 * 0.00833),
            self.game_data.game_tick_packet,
            rendering_enabled=False,
        )

    def get_mechanic_controls(self):
        return self.mechanic.get_controls(self.game_data.my_car, self.game_data.game_tick_packet)
