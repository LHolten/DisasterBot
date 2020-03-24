from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from util.boost_utils import closest_available_boost
from mechanic.drive_navigate_boost import DriveNavigateBoost


class CollectBoost(BaseAction):

    """Simple Action to collect 100 boost"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveNavigateBoost(self.agent, self.rendering_enabled)

    def get_controls(self, game_data) -> SimpleControllerState:

        car = game_data.my_car

        boost_pads = game_data.boost_pads[game_data.boost_pads["is_full_boost"]]
        boost_pad = closest_available_boost(car.location + car.velocity / 2, boost_pads)

        self.finished = car.boost >= 100
        self.failed = boost_pad is None

        return self.mechanic.get_controls(game_data.my_car, game_data.boost_pads, boost_pad["location"])

    def is_valid(self, game_data) -> bool:
        boosts_are_active = len(game_data.boost_pads["is_active"]) > 0
        return boosts_are_active and game_data.my_car.on_ground

    def eta(self, game_data) -> float:
        raise NotImplementedError
