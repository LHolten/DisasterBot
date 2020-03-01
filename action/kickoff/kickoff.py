import math
from rlbot.agents.base_agent import SimpleControllerState
from action.base_action import BaseAction

from mechanic.drive_navigate_boost import DriveNavigateBoost


class Kickoff(BaseAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveNavigateBoost(self.agent, self.rendering_enabled)

    def get_controls(self, game_data) -> SimpleControllerState:

        self.controls = self.mechanic.step(game_data.my_car, game_data.boost_pads, game_data.ball.location)

        ball_loc = game_data.ball.location
        kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 1

        if not kickoff:
            self.finished = True
        else:
            self.finished = False

        return self.controls
