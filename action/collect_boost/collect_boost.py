from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from util.boost_utils import closest_available_boost
from mechanic.drive import DriveTurnToFaceTarget


class CollectBoost(BaseAction):

    def __init__(self, agent):
        super(CollectBoost, self).__init__(agent)
        self.mechanic = None

    def get_controls(self, game_data) -> SimpleControllerState:

        car = game_data.my_car

        if not self.mechanic:
            self.mechanic = DriveTurnToFaceTarget(self.agent)

        boost_pad = closest_available_boost(car.location + car.velocity / 2,
                                            game_data.large_pads + game_data.small_pads)

        if boost_pad is None:
            # All boost pads are inactive.
            return self.controls

        self.mechanic.step(game_data.my_car, boost_pad.location)
        self.controls = self.mechanic.controls

        return self.controls

    def get_possible(self, game_data):
        return True

    def update_status(self, game_data):

        boost = game_data.my_car.boost

        if boost == 100:
            self.finished = True
