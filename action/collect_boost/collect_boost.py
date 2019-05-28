from action.base_action import BaseAction
from util.boost_utils import closest_available_boost
from rlbot.agents.base_agent import SimpleControllerState

from rlutilities.mechanics import Drive
from rlutilities.simulation import Game


class CollectBoost(BaseAction):
    action = None

    def get_output(self, info: Game) -> SimpleControllerState:
        car = info.my_car
        if not self.action:
            self.action = Drive(car)

        boost_pad = closest_available_boost(car.location, info.pads)

        if boost_pad is None:
            # All boost pads are inactive.
            return self.controls

        self.action.target = boost_pad.location

        self.action.step(info.time_delta)
        self.controls = self.action.controls

        return self.controls

    def get_possible(self, info: Game):
        return True

    def update_status(self, info: Game):
        if info.my_car.boost == 100:
            self.finished = True
