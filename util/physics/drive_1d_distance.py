from numba import jit, f8, guvectorize
from numba.types import UniTuple

from util.physics.drive_1d_solutions import (
    State,
    VelocityNegative,
    Velocity0To1400Boost,
    Velocity0To1400,
    Velocity1400To2300,
)


state_distance_step_range_negative = VelocityNegative.wrap_time_travel_distance_step()
state_distance_step_range_0_1400_boost = Velocity0To1400Boost.wrap_time_travel_distance_step()
state_distance_step_range_0_1400 = Velocity0To1400.wrap_time_travel_distance_step()
state_distance_step_range_1400_2300 = Velocity1400To2300.wrap_time_travel_distance_step()


@jit(UniTuple(f8, 3)(f8, f8, f8), nopython=True, fastmath=True)
def state_at_distance(distance: float, initial_velocity: float, boost_amount: float) -> float:
    """Returns the state reached (time, vel, boost)
    after driving forward and using boost and reaching a certain distance."""

    if distance == 0:
        return 0.0, initial_velocity, boost_amount

    state = State(distance, initial_velocity, boost_amount, 0.0)

    state = state_distance_step_range_negative(state)
    state = state_distance_step_range_0_1400_boost(state)
    state = state_distance_step_range_0_1400(state)
    state = state_distance_step_range_1400_2300(state)

    return state.time + state.dist / state.vel, state.vel, state.boost


@guvectorize(["(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])"], "(n), (n), (n) -> (n), (n), (n)", nopython=True)
def state_at_distance_vectorized(
    distance: float, initial_velocity: float, boost_amount: float, out_time, out_vel, out_boost
) -> float:
    """Returns the states reached (time[], vel[], boost[])
    after driving forward and using boost and reaching a certain distance."""
    for i in range(len(distance)):
        out_time[i], out_vel[i], out_boost[i] = state_at_distance(distance[i], initial_velocity[i], boost_amount[i])


def main():

    from timeit import timeit
    import numpy as np

    initial_velocity = np.linspace(-2300, 2300, 360)
    desired_dist = np.linspace(0, 6000, 360)
    boost_amount = np.linspace(0, 100, 360)

    def test_function():
        return state_at_distance_vectorized(desired_dist, initial_velocity, boost_amount)

    print(test_function())

    fps = 120
    n_times = 10000
    time_taken = timeit(test_function, number=n_times)
    percentage = time_taken * fps / n_times * 100

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage:.5f} % of our time budget.")


if __name__ == "__main__":
    main()
