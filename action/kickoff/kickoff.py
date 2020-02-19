import math
from rlbot.agents.base_agent import SimpleControllerState
from action.base_action import BaseAction

from mechanic.drive_arrive_in_time import DriveArriveInTime


class Kickoff(BaseAction):

    mechanic = None

    def get_controls(self, game_data) -> SimpleControllerState:

        if self.mechanic is None:
            self.mechanic = DriveArriveInTime(self.agent, self.rendering_enabled)

        self.mechanic.step(game_data.my_car, game_data.ball.location, 0)
        self.controls = self.mechanic.controls

        ball_loc = game_data.ball.location
        kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 1

        if not kickoff:
            self.finished = True
        else:
            self.finished = False

        return self.controls
