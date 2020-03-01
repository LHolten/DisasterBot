from rlbot.agents.base_agent import SimpleControllerState

from mechanic.base_mechanic import BaseMechanic
from mechanic.drive_arrive_in_time import DriveArriveInTime
from skeleton.util.structure import Player

from util.linear_algebra import norm
from util.path_finder import find_fastest_path, first_target, optional_boost_target


class DriveNavigateBoost(BaseMechanic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveArriveInTime(self.agent, rendering_enabled=self.rendering_enabled)

    def step(self, car: Player, boost_pads, target_loc, target_dt=0) -> SimpleControllerState:
        # path = find_fastest_path(boost_pads, car.location, target_loc, car.velocity, car.boost)
        # target = first_target(boost_pads, target_loc, path)
        target = optional_boost_target(boost_pads, car.location, target_loc, car.velocity, car.boost)

        time = target_dt if (target == target_loc).all() else 0

        # updating status
        if norm(car.location - target_loc) < 25 and abs(target_dt) < 0.05:
            self.finished = True
        else:
            self.finished = False

        return self.mechanic.step(car, target, time)
