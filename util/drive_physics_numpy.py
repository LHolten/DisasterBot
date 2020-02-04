from typing import Dict, Any

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


def get_acceleration_offset(boost: bool):
    return THROTTLE_ACCELERATION_0 + BOOST_ACCELERATION if boost else THROTTLE_ACCELERATION_0


class VelocityRange:
    max_speed = None

    @staticmethod
    def distance_traveled(t: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def velocity_reached(t: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def time_reach_velocity(v: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def state_step(cls, state: Dict, boost: bool):
        """Advances the state to the soonest phase end."""
        mask = state['vel'] < cls.max_speed

        time = cls.time_reach_velocity(cls.max_speed, state['vel'], boost)
        time = np.minimum(time, state['time'])

        if boost:
            time = np.minimum(time, state['boost'] / BOOST_CONSUMPTION_RATE)
            state['boost'] = np.where(mask, state['boost'] - time *
                                      BOOST_CONSUMPTION_RATE, state['boost'])

        state['dist'] = np.where(mask, state['dist'] +
                                 cls.distance_traveled(time, state['vel'], boost), state['dist'])
        state['vel'] = np.where(mask, cls.velocity_reached(time, state['vel'], boost), state['vel'])
        state['time'] = np.where(mask, state['time'] - time, state['time'])


class Velocity0To1400(VelocityRange):
    max_speed = THROTTLE_MID_SPEED

    @staticmethod
    def distance_traveled(t: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        b = get_acceleration_offset(boost)
        sub = np.expm1(a * t)
        return (b * (-a * t + sub) + a * v0 * sub) / np.square(a)

    @staticmethod
    def velocity_reached(t: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        b = get_acceleration_offset(boost)
        return (b * np.expm1(a * t)) / a + v0 * np.exp(a * t)

    @staticmethod
    def time_reach_velocity(v: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        b = get_acceleration_offset(boost)
        return np.log((a * v + b) / (a * v0 + b)) / a


# for when the only acceleration that applies is from boost.
class Velocity1400To2300(Velocity0To1400):
    max_speed = MAX_CAR_SPEED

    @staticmethod
    def distance_traveled(t: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        return t * (BOOST_ACCELERATION * t + 2 * v0) / 2

    @staticmethod
    def velocity_reached(t: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        return BOOST_ACCELERATION * t + v0

    @staticmethod
    def time_reach_velocity(v: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        return (v - v0) / BOOST_ACCELERATION


# for when the velocity is opposite the throttle direction,
# only the breaking acceleration applies, boosting has no effect.
# assuming throttle is positive, flip velocity signs if otherwise.
class VelocityNegative(VelocityRange):
    max_speed = 0

    @staticmethod
    def distance_traveled(t: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        return t * (BREAK_ACCELERATION * t + 2 * v0) / 2

    @staticmethod
    def velocity_reached(t: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        return BREAK_ACCELERATION * t + v0

    @staticmethod
    def time_reach_velocity(v: np.ndarray, v0: np.ndarray, boost: bool) -> np.ndarray:
        return (v - v0) / BREAK_ACCELERATION


def distance_traveled_numpy(t: Any, v0: Any, boost_amount: Any) -> np.ndarray:
    """Returns the max distance driven forward using boost, this allows any starting velocity
    assuming we're not using boost when going backwards and using it otherwise."""
    state = {'time': t, 'vel': v0, 'boost': boost_amount, 'dist': np.zeros_like(t)}

    VelocityNegative.state_step(state, False)
    Velocity0To1400.state_step(state, True)
    Velocity0To1400.state_step(state, False)
    Velocity1400To2300.state_step(state, True)

    return state['dist'] + state['time'] * state['vel']


def main():

    from timeit import timeit

    time = np.array([range(0, 360)]) / 60
    initial_velocity = np.array([range(-180, 180)]) * 12
    boost_amount = np.array([range(0, 360)]) / 360

    def test_function():
        return distance_traveled_numpy(time, initial_velocity, boost_amount)

    print(np.array(test_function(), dtype='object'))

    fps = 120
    n_times = 10000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == '__main__':
    main()
