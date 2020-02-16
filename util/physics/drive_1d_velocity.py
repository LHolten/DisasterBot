from numba import jit, f8, vectorize

from util.physics.drive_1d_solutions import (
    State,
    VelocityNegative,
    Velocity0To1400Boost,
    Velocity0To1400,
    Velocity1400To2300,
)


time_velocity_step_range_negative = VelocityNegative.wrap_time_reach_velocity_step()
time_velocity_step_range_0_1400_boost = Velocity0To1400Boost.wrap_time_reach_velocity_step()
time_velocity_step_range_0_1400 = Velocity0To1400.wrap_time_reach_velocity_step()
time_velocity_step_range_1400_2300 = Velocity1400To2300.wrap_time_reach_velocity_step()


@jit(f8(f8, f8, f8), nopython=True, fastmath=True)
def time_at_velocity(desired_velocity: float, initial_velocity: float, boost_amount: float) -> float:
    """Returns the time it takes to reach any desired velocity including those that require reversing."""
    state = State(desired_velocity, initial_velocity, boost_amount, 0.0)

    state = time_velocity_step_range_negative(state)
    state = time_velocity_step_range_0_1400_boost(state)
    state = time_velocity_step_range_0_1400(state)
    state = time_velocity_step_range_1400_2300(state)

    if state[0] != state.vel:
        return 10.0
    return state.time


time_at_velocity_vectorized = vectorize([f8(f8, f8, f8)], nopython=True)(time_at_velocity)


def main():

    from timeit import timeit
    import numpy as np

    initial_velocity = np.linspace(-2300, 2300, 360)
    desired_vel = -initial_velocity
    boost_amount = np.linspace(0, 100, 360)

    def test_function():
        return time_at_velocity_vectorized(desired_vel, initial_velocity, boost_amount)

    print(test_function())

    fps = 120
    n_times = 10000
    time_taken = timeit(test_function, number=n_times)
    percentage = time_taken * fps / n_times * 100

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage:.5f} % of our time budget.")


if __name__ == "__main__":
    main()
