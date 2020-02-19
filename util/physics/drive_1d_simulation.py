from numba import jit, vectorize, guvectorize, f8
from numba.types import UniTuple

THROTTLE_ACCELERATION_0 = 1600
THROTTLE_ACCELERATION_1400 = 160
THROTTLE_MID_SPEED = 1400

BOOST_ACCELERATION = 991.6667
BREAK_ACCELERATION = 3500
COAST_ACCELERATION = 525

MAX_CAR_SPEED = 2300

BOOST_CONSUMPTION_RATE = 33.3  # per second
BOOST_MIN_TIME = 14 / 120  # The minimum time window where boost takes effect
BOOST_MIN_ACCELERATION = BOOST_ACCELERATION * BOOST_MIN_TIME

# constants of the acceleration between 0 to 1400 velocity: acceleration = a * velocity + b
a = -(THROTTLE_ACCELERATION_0 - THROTTLE_ACCELERATION_1400) / THROTTLE_MID_SPEED
b = THROTTLE_ACCELERATION_0

DT = 1 / 120


@vectorize([f8(f8)], nopython=True, fastmath=True)
def sign(x: float):
    return 1 if x >= 0 else -1


@vectorize([f8(f8, f8)], nopython=True, fastmath=True)
def throttle_acceleration(vel: float, throttle: float = 1):
    throttle = min(1, max(-1, throttle))
    if throttle * vel < 0:
        return -BREAK_ACCELERATION * sign(vel)
    elif throttle == 0:
        return -COAST_ACCELERATION * sign(vel)
    elif abs(vel) < THROTTLE_MID_SPEED:
        return (a * min(abs(vel), THROTTLE_MID_SPEED) + b) * throttle
    else:
        return 0


@jit([UniTuple(f8, 3)(f8, f8, f8)], nopython=True, fastmath=True)
def state_at_distance_simulation(max_distance: float, v_0: float, initial_boost: float):

    time = 0
    distance = 0
    velocity = v_0
    boost = initial_boost

    while distance < max_distance:
        to_boost = 0 <= velocity < MAX_CAR_SPEED and boost > 0
        acceleration = throttle_acceleration(velocity, 1) + to_boost * BOOST_ACCELERATION
        distance = distance + velocity * DT + 0.5 * acceleration * DT * DT
        velocity = min(velocity + acceleration * DT, MAX_CAR_SPEED)
        boost -= to_boost * BOOST_CONSUMPTION_RATE * DT
        time += DT
        if time > 6:
            break

    return time, velocity, max(boost, 0)


@guvectorize(["(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])"], "(n), (n), (n) -> (n), (n), (n)", nopython=True)
def state_at_distance_simulation_vectorized(
    max_distance, initial_velocity, boost_amount, out_time, out_vel, out_boost
) -> float:
    for i in range(len(max_distance)):
        out_time[i], out_vel[i], out_boost[i] = state_at_distance_simulation(
            max_distance[i], initial_velocity[i], boost_amount[i]
        )


@jit([UniTuple(f8, 3)(f8, f8, f8)], nopython=True, fastmath=True)
def state_at_time_simulation(time_window: float, initial_velocity: float, boost_amount: float):

    distance = 0
    velocity = initial_velocity
    boost = boost_amount

    for i in range(round(time_window / DT)):
        to_boost = 0 <= velocity < MAX_CAR_SPEED and boost > 0
        acceleration = throttle_acceleration(velocity, 1) + to_boost * BOOST_ACCELERATION
        distance = distance + velocity * DT + 0.5 * acceleration * DT * DT
        velocity = min(velocity + acceleration * DT, MAX_CAR_SPEED)
        boost -= to_boost * BOOST_CONSUMPTION_RATE * DT

    return distance, velocity, max(boost, 0)


@guvectorize(["(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])"], "(n), (n), (n) -> (n), (n), (n)", nopython=True)
def state_at_time_simulation_vectorized(time, initial_velocity, boost_amount, out_dist, out_vel, out_boost) -> float:
    for i in range(len(time)):
        out_dist[i], out_vel[i], out_boost[i] = state_at_time_simulation(time[i], initial_velocity[i], boost_amount[i])


@jit([UniTuple(f8, 3)(f8, f8, f8)], nopython=True, fastmath=True)
def state_at_velocity_simulation(desired_velocity: float, initial_velocity: float, boost_amount: float):

    time = 0
    distance = 0
    velocity = initial_velocity
    boost = boost_amount

    if desired_velocity > initial_velocity:
        while velocity < desired_velocity:
            to_boost = 0 <= velocity < MAX_CAR_SPEED and boost > 0
            acceleration = throttle_acceleration(velocity, 1) + to_boost * BOOST_ACCELERATION
            distance = distance + velocity * DT + 0.5 * acceleration * DT * DT
            velocity = min(velocity + acceleration * DT, MAX_CAR_SPEED)
            boost -= to_boost * BOOST_CONSUMPTION_RATE * DT
            time += DT
            if time > 6:
                return 10, 10000, boost
    else:
        while velocity > desired_velocity:
            acceleration = throttle_acceleration(velocity, -1)
            distance = distance + velocity * DT + 0.5 * acceleration * DT * DT
            velocity = max(velocity + acceleration * DT, -MAX_CAR_SPEED)
            time += DT
            if time > 6:
                return 10, 10000, boost
    return time, distance, max(boost, 0)


@guvectorize(["(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:])"], "(n), (n), (n) -> (n), (n), (n)", nopython=True)
def state_at_velocity_simulation_vectorized(
    desired_velocity, initial_velocity, boost_amount, out_time, out_dist, out_boost,
) -> float:
    for i in range(len(desired_velocity)):
        out_time[i], out_dist[i], out_boost[i] = state_at_velocity_simulation(
            desired_velocity[i], initial_velocity[i], boost_amount[i]
        )


def main():

    from timeit import timeit

    import numpy as np

    time = np.linspace(0, 6, 360)
    desired_distance = np.linspace(0, 6000, 360)
    initial_velocity = np.linspace(-2300, 2300, 360)
    boost_amount = np.linspace(0, 100, 360)

    desired_velocity = -initial_velocity

    def test_function():
        return state_at_velocity_simulation_vectorized(desired_velocity, initial_velocity, boost_amount)

    print(test_function())

    fps = 120
    n_times = 1000
    time_taken = timeit(test_function, number=n_times)
    percentage = time_taken * fps / n_times * 100

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage:.5f} % of our time budget.")


if __name__ == "__main__":
    main()
