import math
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from mechanic.base_mechanic import BaseMechanic
from util.numerics import clip, sign
from util.render_utils import render_car_text
from util.generator_utils import initialize_generator
from util.linear_algebra import normalize
from skeleton.util.structure import GameData

PI = math.pi


class JumpingShot(BaseMechanic):
    def __init__(self, agent, target_loc, target_time, game_packet, rendering_enabled=False):
        super().__init__(agent, rendering_enabled=rendering_enabled)
        self.target = target_loc
        self.start_time = game_packet.game_info.seconds_elapsed
        self.inputs = [SimpleControllerState(jump=True), SimpleControllerState(jump=False), None]
        delay = target_time - game_packet.game_info.seconds_elapsed
        self.timers = [delay - (1 / 120) * 2, (1 / 120) * 2, 0.25]
        self.done_timer = target_time + self.timers[2]
        print(self.timers[0], self.timers[1], target_loc[2])

    def get_flip_inputs(self, car, game_packet):
        # mechanic expects target's future coordinate so results are deviant in testing when just given ball's expired position
        target_in_local_coords = (self.target - car.location).dot(car.rotation_matrix)
        target_angle = math.atan2(target_in_local_coords[1], target_in_local_coords[0])
        yaw = math.sin(target_angle)
        pitch = -math.cos(target_angle)
        return SimpleControllerState(jump=True, yaw=yaw, pitch=pitch)

    def get_index(self, current_time):
        local_time = current_time - self.start_time
        if local_time < self.timers[0]:
            return 0
        elif local_time < self.timers[0] + self.timers[1]:
            return 1
        else:
            return 2

    def get_controls(self, car, game_packet) -> SimpleControllerState:
        self.controls = self.inputs[self.get_index(game_packet.game_info.seconds_elapsed)]
        if self.controls is None:
            self.inputs[2] = self.get_flip_inputs(car, game_packet)
            self.controls = self.inputs[2]

        self.finished = game_packet.game_info.seconds_elapsed > self.done_timer

        return self.controls
