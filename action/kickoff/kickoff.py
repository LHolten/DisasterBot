import math

from action.base_action import BaseAction
from rlbot.agents.base_agent import SimpleControllerState

from rlutilities.simulation import Game
from rlutilities.linear_algebra import dot, norm


class Kickoff(BaseAction):

    def get_output(self, info: Game) -> SimpleControllerState:

        ball = info.ball
        car = info.my_car

        local_coords = dot(ball.location - car.location, car.rotation)

        self.controls.steer = math.copysign(1.0, local_coords[1])

        # just set the throttle to 1 so the car is always moving forward
        self.controls.throttle = 1.0

        return self.controls

    def get_possible(self, info: Game):
        return True

    def update_status(self, info: Game):

        if norm(info.ball.location) > 140 and norm(info.ball.location) > 9:  # this only works for soccar

            if norm(info.ball.location - info.my_car.location) < 240:
                self.finished = True
            else:
                self.failed = True
