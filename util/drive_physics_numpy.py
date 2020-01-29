from typing import Type

import numpy as np

THROTTLE_ACCEL_0 = 1600
THROTTLE_ACCEL_1400 = 160
THROTTLE_MID_SPEED = 1400

BOOST_ACCEL = 991.6667
BREAK_ACCEL = 3500

MAX_CAR_SPEED = 2300

BOOST_CONSUMPTION_RATE = 33.333  # per second

# constants of the acceleration between 0 to 1400 velocity: acceleration = a * velocity + b
a = - (THROTTLE_ACCEL_0 - THROTTLE_ACCEL_1400) / THROTTLE_MID_SPEED
b = THROTTLE_ACCEL_0


DIST = 0
TIME = 1
VEL = 2
BOOST = 3


class VelocityRange:
    @staticmethod
    def distance_traveled(t, v0, is_boosting):
        raise NotImplementedError

    @staticmethod
    def velocity_reached(t, v0, is_boosting):
        raise NotImplementedError

    @staticmethod
    def time_to_reach_velocity(v, v0, is_boosting):
        raise NotImplementedError

    @staticmethod
    def phase_end(state, is_boosting):
        raise NotImplementedError


class Velocity0To1400(VelocityRange):
    @staticmethod
    def distance_traveled(t, v0, is_boosting):
        b2 = b + is_boosting * BOOST_ACCEL
        return (b2 * (-a * t + np.expm1(a * t)) + a * v0 * np.expm1(a * t)) / np.square(a)

    @staticmethod
    def velocity_reached(t, v0, is_boosting):
        b2 = b + is_boosting * BOOST_ACCEL
        return (b2 * np.expm1(a * t)) / a + v0 * np.exp(a * t)

    @staticmethod
    def time_to_reach_velocity(v, v0, is_boosting):
        b2 = b + is_boosting * BOOST_ACCEL
        return np.log((a * v + b2) / (a * v0 + b2)) / a

    @staticmethod
    def phase_end(state, is_boosting):
        time_1400_vel = Velocity0To1400.time_to_reach_velocity(
            THROTTLE_MID_SPEED, state[VEL], is_boosting)

        return np.where(is_boosting,
                        np.minimum(time_to_reach_0_boost(state[BOOST]), time_1400_vel),
                        time_1400_vel)


# for when the only acceleration that applies is from boost.
class Velocity1400To2300Boost(VelocityRange):
    @staticmethod
    def distance_traveled(t, v0, _):
        return t * (BOOST_ACCEL * t + 2 * v0) / 2

    @staticmethod
    def velocity_reached(t, v0, _):
        return BOOST_ACCEL * t + v0

    @staticmethod
    def time_to_reach_velocity(v, v0, _):
        return (v - v0) / BOOST_ACCEL

    @staticmethod
    def phase_end(state, _):
        time_0_boost = time_to_reach_0_boost(state[BOOST])
        time_2300_vel = Velocity1400To2300Boost.time_to_reach_velocity(
            MAX_CAR_SPEED, state[VEL], None)
        return np.minimum(time_0_boost, time_2300_vel)


# for when the velocity is opposite the throttle direction,
# only the breaking acceleration applies, boosting has no effect.
# assuming throttle is positive, flip velocity signs if otherwise.
class VelocityNegative(VelocityRange):
    @staticmethod
    def distance_traveled(t, v0, _):
        return t * (BREAK_ACCEL * t + 2 * v0) / 2

    @staticmethod
    def velocity_reached(t, v0, _):
        return BREAK_ACCEL * t + v0

    @staticmethod
    def time_to_reach_velocity(v, v0, _):
        return (v - v0) / BREAK_ACCEL

    @staticmethod
    def phase_end(state, _):
        return VelocityNegative.time_to_reach_velocity(0, state[VEL], None)


def distance_traveled_zero_acceleration(t, v0):
    return t * v0


# boost consumption
def time_to_reach_0_boost(boost_amount):
    return boost_amount / BOOST_CONSUMPTION_RATE


def boost_reached(time, initial_boost_amount):
    return np.maximum(0, initial_boost_amount - time * BOOST_CONSUMPTION_RATE)


def state_step(state, vel_range: Type[VelocityRange], boost):
    """Advances the state to the soonest phase end."""
    state_copy = np.array(state, copy=True)
    time = np.minimum(vel_range.phase_end(state_copy, boost), state_copy[TIME])
    state_copy[DIST] = state[DIST] + vel_range.distance_traveled(time, state[VEL], boost)
    state_copy[VEL] = vel_range.velocity_reached(time, state[VEL], boost)
    state_copy[TIME] = state[TIME] - time
    state_copy[BOOST] = np.where(boost, boost_reached(time, state[BOOST]), state[BOOST])
    return state_copy


# this allows any starting velocity
def distance_traveled_numpy(t, v0, boost_amount):

    zeros = np.zeros_like(t)
    state = np.array([zeros, t, v0, boost_amount], copy=True)

    state = np.where(state[VEL] < 0,
                     state_step(state, VelocityNegative, False),
                     state)

    state = np.where((state[VEL] < THROTTLE_MID_SPEED) & (state[BOOST] > 0),
                     state_step(state, Velocity0To1400, True),
                     state)

    state = np.where((state[VEL] < THROTTLE_MID_SPEED) & (state[BOOST] <= 0),
                     state_step(state, Velocity0To1400, False),
                     state)

    state = np.where((state[VEL] < MAX_CAR_SPEED) & (state[BOOST] > 0),
                     state_step(state, Velocity1400To2300Boost, False),
                     state)

    return state[DIST] + distance_traveled_zero_acceleration(state[TIME], state[VEL])


def main():

    from timeit import timeit

    time = np.array([1.821725257142, 2] * 2)
    initial_velocity = np.array([-2300, -400] * 2)
    boost_amount = np.array([0, 30] * 2)

    def test_function():
        return distance_traveled_numpy(time, initial_velocity, boost_amount)

    print(np.array(test_function(), dtype='object'))

    fps = 120
    n_times = 1000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == '__main__':
    main()
