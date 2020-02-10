import math
import numpy as np
from numba import vectorize, f8

from util.special_lambertw import lambertw

THROTTLE_ACCELERATION_0 = 1600
THROTTLE_ACCELERATION_1400 = 160
THROTTLE_MID_SPEED = 1400

BOOST_ACCELERATION = 991.6667
BREAK_ACCELERATION = 3500
COAST_ACCELERATION = 525

MAX_CAR_SPEED = 2300

BOOST_CONSUMPTION_RATE = 33.3  # per second

# constants of the acceleration between 0 to 1400 velocity: acceleration = a * velocity + b
a = -(THROTTLE_ACCELERATION_0 - THROTTLE_ACCELERATION_1400) / THROTTLE_MID_SPEED
b = THROTTLE_ACCELERATION_0
b2 = b + BOOST_ACCELERATION  # when boosting, we simply add the boost acceleration


# driving within the 0 to 1400 velocity range with boost
@vectorize([f8(f8, f8)])
def distance_traveled_0_1400(t: float, v0: float):
    return (b * (-a * t + math.expm1(a * t)) + a * v0 * math.expm1(a * t)) / math.pow(a, 2)


@vectorize([f8(f8, f8)])
def velocity_reached_0_1400(t: float, v0: float):
    return (b * math.expm1(a * t)) / a + v0 * math.exp(a * t)


@vectorize([f8(f8, f8)])
def time_reach_velocity_0_1400(v: float, v0: float):
    return math.log((a * v + b) / (a * v0 + b)) / a


@vectorize([f8(f8, f8)])
def time_travel_distance_0_1400(d: float, v: float):
    return (-d * a * a - b * lambertw(-((b + a * v) * math.exp(-(a * (v + a * d)) / b - 1)) / b) - a * v - b) / (a * b)


# driving within the 0 to 1400 velocity range with boost
@vectorize([f8(f8, f8)])
def distance_traveled_0_1400_Boost(t: float, v0: float):
    return (b2 * (-a * t + math.expm1(a * t)) + a * v0 * math.expm1(a * t)) / math.pow(a, 2)


@vectorize([f8(f8, f8)])
def velocity_reached_0_1400_Boost(t: float, v0: float):
    return (b2 * math.expm1(a * t)) / a + v0 * math.exp(a * t)


@vectorize([f8(f8, f8)])
def time_reach_velocity_0_1400_Boost(v: float, v0: float):
    return math.log((a * v + b2) / (a * v0 + b2)) / a


@vectorize([f8(f8, f8)])
def time_travel_distance_0_1400_Boost(d: float, v: float):
    return (-d * a * a - b2 * lambertw(-((b2 + a * v) * math.exp(-(a * (v + a * d)) / b - 1)) / b2) - a * v - b2) / (
        a * b2
    )


# for when the only acceleration that applies is from boost.
@vectorize([f8(f8, f8)])
def distance_traveled_1400_2300_boost(t: float, v0: float):
    return t * (BOOST_ACCELERATION * t + 2 * v0) / 2


@vectorize([f8(f8, f8)])
def velocity_reached_1400_2300_boost(t: float, v0: float):
    return BOOST_ACCELERATION * t + v0


@vectorize([f8(f8, f8)])
def time_reach_velocity_1400_2300_boost(v: float, v0: float):
    return (v - v0) / BOOST_ACCELERATION


@vectorize([f8(f8, f8)])
def time_reach_distance_1400_2300_boost(d: float, v: float):
    return -(math.sqrt(2 * BOOST_ACCELERATION * d + math.pow(v, 2)) + v) / a


@vectorize([f8(f8, f8)])
def time_travel_distance_1400_2300(d: float, v: float):
    return (-v + math.sqrt(2 * BOOST_ACCELERATION * d + math.pow(v, 2))) / BOOST_ACCELERATION


# for when the velocity is opposite the throttle direction,
# only the breaking acceleration applies, boosting has no effect.
# assuming throttle is positive, flip velocity signs if otherwise.
@vectorize([f8(f8, f8)])
def distance_traveled_negative(t: float, v0: float):
    return t * (BREAK_ACCELERATION * t + 2 * v0) / 2


@vectorize([f8(f8, f8)])
def velocity_reached_negative(t: float, v0: float):
    return BREAK_ACCELERATION * t + v0


@vectorize([f8(f8, f8)])
def time_reach_velocity_negative(v: float, v0: float):
    return (v - v0) / BREAK_ACCELERATION


@vectorize([f8(f8, f8)])
def time_reach_distance_negative(d: float, v: float):
    return -(math.sqrt(2 * BREAK_ACCELERATION * d + math.pow(v, 2)) + v) / a


@vectorize([f8(f8, f8)])
def time_travel_distance_negative(d: float, v: float):
    return (-v + math.sqrt(2 * BREAK_ACCELERATION * d + math.pow(v, 2))) / BREAK_ACCELERATION


# no acceleration
@vectorize([f8(f8, f8)])
def distance_traveled_zero_acceleration(t: float, v0: float):
    return t * v0


@vectorize([f8(f8, f8)])
def time_travel_distance_zero_acceleration(d: float, v0: float):
    return d / v0


# boost consumption
@vectorize([f8(f8)])
def time_reach_0_boost(boost_amount: float):
    return boost_amount / BOOST_CONSUMPTION_RATE


@vectorize([f8(f8, f8)])
def boost_reached(time: float, initial_boost_amount: float):
    return max(0, initial_boost_amount - time * BOOST_CONSUMPTION_RATE)


@vectorize([f8(f8, f8, f8)])
def distance_traveled(time_window: float, initial_velocity: float, boost_amount: float):
    """Returns the max distance driven forward using boost, this allows any starting velocity
      assuming we're not using boost when going backwards and using it otherwise."""

    distance = 0
    time_left = time_window
    velocity = initial_velocity
    boost_left = boost_amount

    if velocity < 0:

        time_0_vel = time_reach_velocity_negative(0, velocity)
        if time_left <= time_0_vel:
            return distance + distance_traveled_negative(time_left, velocity)
        else:
            distance = distance + distance_traveled_negative(time_0_vel, velocity)
            time_left = time_left - time_0_vel
            velocity = 0

    if velocity < THROTTLE_MID_SPEED:

        if boost_left > 0:
            time_0_boost = time_reach_0_boost(boost_left)
            time_1400_vel = time_reach_velocity_0_1400_Boost(THROTTLE_MID_SPEED, velocity)

            if time_0_boost <= time_1400_vel:
                if time_left <= time_0_boost:
                    return distance + distance_traveled_0_1400_Boost(time_left, velocity)
                else:
                    distance = distance + distance_traveled_0_1400_Boost(time_0_boost, velocity)
                    velocity = velocity_reached_0_1400_Boost(time_0_boost, velocity)
                    time_left = time_left - time_0_boost
                    boost_left = 0
            else:
                if time_left <= time_1400_vel:
                    return distance + distance_traveled_0_1400_Boost(time_left, velocity)
                else:
                    distance = distance + distance_traveled_0_1400_Boost(time_1400_vel, velocity)
                    boost_left = boost_reached(time_1400_vel, boost_left)
                    time_left = time_left - time_1400_vel
                    velocity = THROTTLE_MID_SPEED

        if boost_left <= 0:
            time_1400_vel = time_reach_velocity_0_1400(THROTTLE_MID_SPEED, velocity)
            if time_left <= time_1400_vel:
                return distance + distance_traveled_0_1400(time_left, velocity)
            else:
                distance = distance + distance_traveled_0_1400(time_1400_vel, velocity)
                boost_left = boost_reached(time_1400_vel, boost_left)
                time_left = time_left - time_1400_vel
                velocity = THROTTLE_MID_SPEED

    if velocity < MAX_CAR_SPEED:

        if boost_left > 0:
            time_0_boost = time_reach_0_boost(boost_left)
            time_2300_vel = time_reach_velocity_1400_2300_boost(MAX_CAR_SPEED, velocity)

            if time_0_boost <= time_2300_vel:
                if time_left <= time_0_boost:
                    return distance + distance_traveled_1400_2300_boost(time_left, velocity)
                else:
                    distance = distance + distance_traveled_1400_2300_boost(time_0_boost, velocity)
                    velocity = velocity_reached_1400_2300_boost(time_0_boost, velocity)
                    time_left = time_left - time_0_boost
                    boost_left = 0
            else:
                if time_left <= time_2300_vel:
                    return distance + distance_traveled_1400_2300_boost(time_left, velocity)
                else:
                    distance = distance + distance_traveled_1400_2300_boost(time_2300_vel, velocity)
                    boost_left = boost_reached(time_2300_vel, boost_left)
                    time_left = time_left - time_2300_vel
                    velocity = MAX_CAR_SPEED

    return distance + distance_traveled_zero_acceleration(time_left, velocity)


@vectorize([f8(f8, f8, f8)])
def time_reach_velocity(desired_velocity: float, initial_velocity: float, boost_amount: float):

    time = 0
    boost_left = boost_amount
    velocity = initial_velocity

    if velocity < desired_velocity:

        if velocity < 0:
            if desired_velocity <= 0:
                return time + time_reach_velocity_negative(desired_velocity, velocity)
            else:
                time += time_reach_velocity_negative(0, velocity)
                velocity = 0

        if velocity <= THROTTLE_MID_SPEED:

            if boost_left > 0:
                time_0_boost = time_reach_0_boost(boost_left)
                time_1400_vel = time_reach_velocity_0_1400_Boost(THROTTLE_MID_SPEED, velocity)

                if time_0_boost <= time_1400_vel:

                    velocity_at_0_boost = velocity_reached_0_1400_Boost(time_0_boost, velocity)
                    if desired_velocity <= velocity_at_0_boost:
                        return time + time_reach_velocity_0_1400_Boost(desired_velocity, velocity)
                    else:
                        time += time_0_boost
                        velocity = velocity_at_0_boost
                        boost_left = 0
                else:
                    if desired_velocity <= THROTTLE_MID_SPEED:
                        return time + time_reach_velocity_0_1400_Boost(desired_velocity, velocity)
                    else:
                        time += time_1400_vel
                        velocity = THROTTLE_MID_SPEED
                        boost_left = boost_reached(time_1400_vel, boost_left)

            if boost_left <= 0:
                if desired_velocity <= THROTTLE_MID_SPEED:
                    return time + time_reach_velocity_0_1400(desired_velocity, velocity)

        if desired_velocity <= MAX_CAR_SPEED:

            time_0_boost = time_reach_0_boost(boost_left)
            velocity_at_0_boost = velocity_reached_1400_2300_boost(time_0_boost, velocity)

            if desired_velocity <= velocity_at_0_boost:
                return time + time_reach_velocity_1400_2300_boost(desired_velocity, velocity)

        # Since we don't have any means of acceleration to reach higher velocities
        # we can return a high value or +infinity
        return 10

    elif desired_velocity < velocity:

        if 0 < velocity:
            if 0 < desired_velocity:
                return time + time_reach_velocity_negative(-desired_velocity, -velocity)
            else:
                time = time + time_reach_velocity_negative(0, -velocity)
                velocity = 0

        if -THROTTLE_MID_SPEED < velocity:

            if -THROTTLE_MID_SPEED <= desired_velocity:
                return time + time_reach_velocity_0_1400(-desired_velocity, -velocity)

        return 10
    else:
        return 0


def main():

    from timeit import timeit

    time = np.linspace(0, 6, 360)
    initial_velocity = np.linspace(-2300, 2300, 360)
    boost_amount = np.linspace(0, 100, 360)

    def test_function():
        return distance_traveled(time, initial_velocity, boost_amount)

    print(test_function())

    fps = 120
    n_times = 10000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == "__main__":
    main()
