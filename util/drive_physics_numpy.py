from typing import Type

import numpy as np

THROTTLE_ACCELERATION_0 = 1600
THROTTLE_ACCELERATION_1400 = 160
THROTTLE_MID_SPEED = 1400

BOOST_ACCELERATION = 991.6667
BREAK_ACCELERATION = 3500

MAX_CAR_SPEED = 2300

BOOST_CONSUMPTION_RATE = 33.333  # per second

# constants of the acceleration between 0 to 1400 velocity: acceleration = a * velocity + b
a = - (THROTTLE_ACCELERATION_0 - THROTTLE_ACCELERATION_1400) / THROTTLE_MID_SPEED


def get_b(boost):
    return THROTTLE_ACCELERATION_0 + BOOST_ACCELERATION if boost else THROTTLE_ACCELERATION_0


class VelocityRange:
    @staticmethod
    def distance_traveled(t, v0, boost):
        raise NotImplementedError

    @staticmethod
    def velocity_reached(t, v0, boost):
        raise NotImplementedError

    @staticmethod
    def time_to_reach_velocity(v, v0, boost):
        raise NotImplementedError

    @staticmethod
    def max_speed():
        raise NotImplementedError


class Velocity0To1400(VelocityRange):
    @staticmethod
    def distance_traveled(t, v0, boost):
        b = get_b(boost)
        return (b * (-a * t + np.expm1(a * t)) + a * v0 * np.expm1(a * t)) / np.square(a)

    @staticmethod
    def velocity_reached(t, v0, boost):
        b = get_b(boost)
        return (b * np.expm1(a * t)) / a + v0 * np.exp(a * t)

    @staticmethod
    def time_to_reach_velocity(v, v0, boost):
        b = get_b(boost)
        return np.log((a * v + b) / (a * v0 + b)) / a

    @staticmethod
    def max_speed():
        return THROTTLE_MID_SPEED


# for when the only acceleration that applies is from boost.
class Velocity1400To2300(Velocity0To1400):
    @staticmethod
    def distance_traveled(t, v0, boost):
        return t * (BOOST_ACCELERATION * t + 2 * v0) / 2

    @staticmethod
    def velocity_reached(t, v0, boost):
        return BOOST_ACCELERATION * t + v0

    @staticmethod
    def time_to_reach_velocity(v, v0, boost):
        return (v - v0) / BOOST_ACCELERATION

    @staticmethod
    def max_speed():
        return MAX_CAR_SPEED


# for when the velocity is opposite the throttle direction,
# only the breaking acceleration applies, boosting has no effect.
# assuming throttle is positive, flip velocity signs if otherwise.
class VelocityNegative(VelocityRange):
    @staticmethod
    def distance_traveled(t, v0, boost):
        return t * (BREAK_ACCELERATION * t + 2 * v0) / 2

    @staticmethod
    def velocity_reached(t, v0, boost):
        return BREAK_ACCELERATION * t + v0

    @staticmethod
    def time_to_reach_velocity(v, v0, boost):
        return (v - v0) / BREAK_ACCELERATION

    @staticmethod
    def max_speed():
        return 0


def state_step(state, vel_range: Type[VelocityRange], boost):
    """Advances the state to the soonest phase end."""
    mask = state['vel'] < vel_range.max_speed()

    time = vel_range.time_to_reach_velocity(vel_range.max_speed(), state['vel'][mask], boost)
    time = np.minimum(time, state['time'][mask])

    if boost:
        time = np.minimum(time, state['boost'][mask] / BOOST_CONSUMPTION_RATE)
        state['boost'][mask] = state['boost'][mask] - time * BOOST_CONSUMPTION_RATE

    state['dist'][mask] = state['dist'][mask] + vel_range.distance_traveled(time, state['vel'][mask], boost)
    state['vel'][mask] = vel_range.velocity_reached(time, state['vel'][mask], boost)
    state['time'][mask] = state['time'][mask] - time


# this allows any starting velocity
def distance_traveled_numpy(t, v0, boost_amount):
    dtype = [('time', float), ('vel', float), ('boost', float), ('dist', float)]
    state = np.empty_like(t, dtype)
    state['time'] = t
    state['vel'] = v0
    state['boost'] = boost_amount
    state['dist'] = 0

    state_step(state, VelocityNegative, False),

    state_step(state, Velocity0To1400, True)

    state_step(state, Velocity0To1400, False)

    state_step(state, Velocity1400To2300, True)

    return state['dist'] + state['time'] * state['vel']


def main():

    from timeit import timeit

    time = np.array([1.821725257142, 2] * 2)
    initial_velocity = np.array([-2300, -400] * 2)
    boost_amount = np.array([0, 30] * 2)

    def test_function():
        return distance_traveled_numpy(time, initial_velocity, boost_amount)

    print(np.array(test_function(), dtype='object'))

    fps = 120
    n_times = 100000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == '__main__':
    main()
