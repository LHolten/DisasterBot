from numba import jit, f8, guvectorize
from numba.types import UniTuple

from util.physics.drive_1d_solutions import (
    State,
    VelocityNegative,
    Velocity0To1400Boost,
    Velocity0To1400,
    Velocity1400To2300,
)


distance_step_range_negative = VelocityNegative.wrap_distance_state_step()
distance_step_range_0_1400_boost = Velocity0To1400Boost.wrap_distance_state_step()
distance_step_range_0_1400 = Velocity0To1400.wrap_distance_state_step()
distance_step_range_1400_2300 = Velocity1400To2300.wrap_distance_state_step()


@jit(UniTuple(f8, 3)(f8, f8, f8), nopython=True, fastmath=True)
def state_at_time(time: float, initial_velocity: float, boost_amount: float) -> float:
    """Returns the state reached after (dist, vel, boost)
    after driving forward and using boost and reaching a certain time."""
    if time == 0.0:
        return 0.0, initial_velocity, boost_amount

    state = State(0.0, initial_velocity, boost_amount, time)

    state = distance_step_range_negative(state)
    state = distance_step_range_0_1400_boost(state)
    state = distance_step_range_0_1400(state)
    state = distance_step_range_1400_2300(state)

    return state.dist + state.time * state.vel, state.vel, state.boost


@guvectorize(["(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])"], "(n), (n), (n) -> (n), (n), (n)", nopython=True)
def state_at_time_vectorized(
    time: float, initial_velocity: float, boost_amount: float, out_dist, out_vel, out_boost
) -> float:
    """Returns the states reached (dist[], vel[], boost[]) after driving forward and using boost."""
    for i in range(len(time)):
        out_dist[i], out_vel[i], out_boost[i] = state_at_time(time[i], initial_velocity[i], boost_amount[i])


def main():

    from timeit import timeit
    import numpy as np

    time = np.linspace(0, 6, 360)
    initial_velocity = np.linspace(-2300, 2300, 360)
    boost_amount = np.linspace(0, 100, 360)

    def test_function():
        return state_at_time_vectorized(time, initial_velocity, boost_amount)[0]

    print(test_function())

    fps = 120
    n_times = 10000
    time_taken = timeit(test_function, number=n_times)
    percentage = time_taken * fps / n_times * 100

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage:.5f} % of our time budget.")


if __name__ == "__main__":
    main()
