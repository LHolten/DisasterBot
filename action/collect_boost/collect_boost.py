from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from util.boost_utils import closest_available_boost
from mechanic.drive_turn_face_target import DriveTurnFaceTarget


class CollectBoost(BaseAction):

    mechanic = None

    def get_controls(self, game_data) -> SimpleControllerState:

        car = game_data.my_car

        if self.mechanic is None:
            self.mechanic = DriveTurnFaceTarget(self.agent, self.rendering_enabled)

        boost_pads = game_data.boost_pads[game_data.boost_pads["is_full_boost"]]
        boost_pad = closest_available_boost(car.location + car.velocity / 2, boost_pads)

        if boost_pad is None:
            # All boost pads are inactive.
            return self.controls

        self.mechanic.step(game_data.my_car, boost_pad["location"])
        self.controls = self.mechanic.controls

        if car.boost == 100:
            self.finished = True
        else:
            self.finished = False

        return self.controls

    def is_valid(self, game_data):
        return len(game_data.boost_pads["is_active"]) > 0
