from rlbot.agents.base_agent import SimpleControllerState

from mechanic.base_mechanic import BaseMechanic
from mechanic.drive_arrive_in_time import DriveArriveInTime
from skeleton.util.structure import Player

from util.linear_algebra import norm
from util.path_finder import optional_boost_target


class DriveNavigateBoost(BaseMechanic):

    """Drive mechanic that picks up boost pads that are helpful on the way to the target."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveArriveInTime(self.agent, self.rendering_enabled)

    def get_controls(self, car: Player, boost_pads, target_loc, target_dt=0) -> SimpleControllerState:
        target = optional_boost_target(boost_pads, car.location, target_loc, car.velocity, car.boost)

        time = target_dt if (target == target_loc).all() else 0

        # updating status
        if norm(car.location - target_loc) < 25 and abs(target_dt) < 0.05:
            self.finished = True
        self.failed = target_dt < -0.05

        return self.mechanic.get_controls(car, target, time)

    def is_valid(self, car: Player, boost_pads, target_loc, target_dt=0):
        raise NotImplementedError

    def eta(self, car: Player, boost_pads, target_loc, target_dt=0):
        raise NotImplementedError
