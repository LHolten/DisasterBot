from numba.pycc import CC

from util.physics.drive_1d_distance import state_at_distance
from util.physics.drive_1d_time import state_at_time
from util.physics.drive_1d_velocity import state_at_velocity

if __name__ == "__main__":
    cc = CC("drive_1d_distance")
    cc.export("state_at_distance", "UniTuple(f8, 3)(f8, f8, f8)")(state_at_distance)
    cc.compile()

    cc = CC("drive_1d_time")
    cc.export("state_at_time", "UniTuple(f8, 3)(f8, f8, f8)")(state_at_time)
    cc.compile()

    cc = CC("drive_1d_velocity")
    cc.export("state_at_velocity", "UniTuple(f8, 3)(f8, f8, f8)")(state_at_velocity)
    cc.compile()
