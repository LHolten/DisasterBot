import math
from rlbot.agents.base_agent import SimpleControllerState
from action.base_action import BaseAction

from mechanic.drive_navigate_boost import DriveNavigateBoost
from mechanic.flip import Flip
from util.generator_utils import initialize_generator
from util.linear_algebra import norm, dot, normalize
import numpy as np


class Kickoff(BaseAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveNavigateBoost(self.agent, self.rendering_enabled)
        self.flip = Flip(self.agent, self.rendering_enabled)
        self.kickoff = self.kickoff_generator()

    def get_controls(self, game_data) -> SimpleControllerState:
        relative_ball = game_data.ball.location - game_data.my_car.location
        if norm(relative_ball) < 800:
            self.controls = self.flip.get_controls(game_data.my_car, relative_ball)
        else:
            self.controls = self.kickoff.send(game_data)

        ball_loc = game_data.ball.location
        kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 1

        if not kickoff:
            self.finished = True
        else:
            self.finished = False

        return self.controls

    @initialize_generator
    def kickoff_generator(self):
        game_data = yield

        while True:
            relative_ball = game_data.ball.location - game_data.my_car.location

            if game_data.my_car.boost > 15 or norm(relative_ball) < 2200:
                if game_data.my_car.location[0] > game_data.ball.location[0]:
                    offset = np.array([-150, 0, 0])
                else:
                    offset = np.array([150, 0, 0])
                game_data = yield self.mechanic.get_controls(
                    game_data.my_car, game_data.boost_pads, game_data.ball.location + offset
                )
            else:
                flip = Flip(self.agent, self.rendering_enabled)
                while not flip.finished:
                    controls = flip.get_controls(game_data.my_car, relative_ball)
                    if dot(game_data.my_car.rotation_matrix[:, 0], normalize(relative_ball)) > 0.5:
                        controls.boost = True
                    game_data = yield controls
