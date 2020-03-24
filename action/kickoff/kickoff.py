import math
from rlbot.agents.base_agent import SimpleControllerState
from action.base_action import BaseAction

from mechanic.drive_navigate_boost import DriveNavigateBoost


class Kickoff(BaseAction):

    """Action to drive to the ball on kickoff (for now)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveNavigateBoost(self.agent, self.rendering_enabled)

    def get_controls(self, game_data) -> SimpleControllerState:

        self.finished = not self.is_valid(game_data)

        return self.mechanic.get_controls(game_data.my_car, game_data.boost_pads, game_data.ball.location, 0)

    def is_valid(self, game_data) -> bool:
        ball_loc = game_data.ball.location
        is_kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 1
        return is_kickoff

    def eta(self, game_data) -> bool:
        return self.mechanic.eta(game_data.my_car, game_data.boost_pads, game_data.ball.location, 0)
