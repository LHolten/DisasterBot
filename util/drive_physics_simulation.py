from numba import vectorize, f8

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


@vectorize([f8(f8)], nopython=True, cache=True)
def sign(x: float):
    return 1 if x >= 0 else -1


@vectorize([f8(f8, f8)], nopython=True, cache=True)
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


@vectorize([f8(f8, f8, f8)], nopython=True, cache=True)
def min_travel_time_simulation(max_distance: float, v_0: float, initial_boost: float):

    DT = 1 / 120
    time = 0
    distance = 0
    velocity = v_0
    boost = initial_boost
    acceleration = 0

    while distance < max_distance:
        to_boost = sign(velocity) and boost > 0
        acceleration = throttle_acceleration(velocity, 1) + to_boost * BOOST_ACCELERATION
        velocity = min(velocity + acceleration * DT, MAX_CAR_SPEED)
        distance = distance + velocity * DT + 0.5 * acceleration * DT * DT
        boost -= to_boost * BOOST_CONSUMPTION_RATE * DT
        time += DT
        if time > 6:
            break

    return time


@vectorize([f8(f8, f8, f8)], nopython=True, cache=True)
def distance_traveled_simulation(time_window: float, initial_velocity: float, boost_amount: float):

    DT = 1 / 120
    distance = 0
    velocity = initial_velocity
    boost = boost_amount
    acceleration = 0

    for i in range(round(time_window / DT)):
        to_boost = velocity >= 0 and boost > 0
        acceleration = throttle_acceleration(velocity, 1) + to_boost * BOOST_ACCELERATION
        velocity = min(velocity + acceleration * DT, MAX_CAR_SPEED)
        distance = distance + velocity * DT + 0.5 * acceleration * DT * DT
        boost -= to_boost * BOOST_CONSUMPTION_RATE * DT

    return distance


@vectorize([f8(f8, f8, f8)], nopython=True, cache=True)
def time_reach_velocity_simulation(desired_velocity: float, initial_velocity: float, boost_amount: float):

    DT = 1 / 120
    time = 0
    velocity = initial_velocity
    boost = boost_amount

    if desired_velocity > initial_velocity:
        while velocity < desired_velocity:
            to_boost = velocity >= 0 and boost > 0
            acceleration = throttle_acceleration(velocity, 1) + to_boost * BOOST_ACCELERATION
            velocity = min(velocity + acceleration * DT, MAX_CAR_SPEED)
            boost -= to_boost * BOOST_CONSUMPTION_RATE * DT
            time += DT
            if time > 6:
                return 10
    else:
        while velocity > desired_velocity:
            acceleration = throttle_acceleration(velocity, -1)
            velocity = max(velocity + acceleration * DT, -MAX_CAR_SPEED)
            time += DT
            if time > 6:
                return 10
    return time


def main():

    from timeit import timeit

    # import numpy as np

    # time = np.linspace(0, 6, 360)
    # initial_velocity = np.linspace(-2300, 2300, 360)
    # boost_amount = np.linspace(0, 100, 360)

    time = 6
    initial_velocity = -2300
    boost_amount = 100

    def test_function():
        return distance_traveled_simulation(time, initial_velocity, boost_amount)

    print(test_function())

    fps = 120
    n_times = 1000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == "__main__":
    main()
