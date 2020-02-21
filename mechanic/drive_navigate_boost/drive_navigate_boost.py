from rlbot.agents.base_agent import SimpleControllerState

from mechanic.base_mechanic import BaseMechanic
from mechanic.drive_arrive_in_time import DriveArriveInTime
from skeleton.util.structure import Player

from util.linear_algebra import norm
from util.path_finder import find_fastest_path, first_target


class DriveNavigateBoost(BaseMechanic):

    turn_mechanic = None

    def step(self, car: Player, boost_pads, target_loc) -> SimpleControllerState:

        if self.turn_mechanic is None:
            self.turn_mechanic = DriveArriveInTime(self.agent, rendering_enabled=True)

        path = find_fastest_path(boost_pads, car.location, target_loc, norm(car.velocity), car.boost)
        target = first_target(boost_pads, target_loc, path)

        # updating status
        if norm(car.location - target_loc) < 40:
            self.finished = True
        else:
            self.finished = False

        return self.turn_mechanic.step(car, target, 0)
