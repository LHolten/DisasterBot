import math
from rlbot.agents.base_agent import SimpleControllerState
from action.base_action import BaseAction

from mechanic.drive_turn_face_target import DriveTurnFaceTarget


class Kickoff(BaseAction):

    def __init__(self, agent):
        super(Kickoff, self).__init__(agent)
        self.mechanic = None

    def get_controls(self, game_data) -> SimpleControllerState:

        if not self.mechanic:
            self.mechanic = DriveTurnFaceTarget(self.agent)

        self.mechanic.step(game_data.my_car, game_data.ball.location)
        self.controls = self.mechanic.controls

        return self.controls

    def update_status(self, game_data):
        ball_loc = game_data.ball.location
        kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 140

        if not kickoff:
            self.finished = True
        else:
            self.finished = False
