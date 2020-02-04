import math
from collections import namedtuple

from numba import vectorize, float64, jit

THROTTLE_ACCELERATION_0 = 1600.
THROTTLE_ACCELERATION_1400 = 160.
THROTTLE_MID_SPEED = 1400.

BOOST_ACCELERATION = 991.6667
BREAK_ACCELERATION = 3500.

MAX_CAR_SPEED = 2300.

BOOST_CONSUMPTION_RATE = 33.333  # per second

# constants of the acceleration between 0 to 1400 velocity: acceleration = a * velocity + b
a = - (THROTTLE_ACCELERATION_0 - THROTTLE_ACCELERATION_1400) / THROTTLE_MID_SPEED


State = namedtuple('State', ['vel', 'boost', 'time', 'dist'])


class VelocityRange:
    max_speed = None
    use_boost = None

    @staticmethod
    def distance_traveled(t: float, v0: float) -> float:
        raise NotImplementedError

    @staticmethod
    def velocity_reached(t: float, v0: float) -> float:
        raise NotImplementedError

    @staticmethod
    def time_reach_velocity(v: float, v0: float) -> float:
        raise NotImplementedError

    @classmethod
    def wrap(cls):
        """Advances the state to the soonest phase end."""

        max_speed = cls.max_speed
        distance_traveled = cls.distance_traveled
        velocity_reached = cls.velocity_reached
        time_reach_velocity = cls.time_reach_velocity

        @jit
        def process_boost(state: State) -> State:
            if max_speed <= state.vel or state.time == 0.:
                return state

            t = min(state.boost / BOOST_CONSUMPTION_RATE, time_reach_velocity(max_speed, state.vel))

            if state.time <= t:
                return State(0., 0., 0., state.dist + distance_traveled(state.time, state.vel))

            vel = velocity_reached(t, state.vel)
            dist = state.dist + distance_traveled(t, state.vel)
            boost = state.boost - t * BOOST_CONSUMPTION_RATE
            time = state.time - t

            return State(vel, boost, time, dist)

        @jit
        def process_no_boost(state: State) -> State:
            if max_speed <= state.vel or state.time == 0.:
                return state

            t = time_reach_velocity(max_speed, state.vel)

            if state.time <= t:
                return State(0., 0., 0., state.dist + distance_traveled(state.time, state.vel))

            dist = state.dist + distance_traveled(t, state.vel)
            time = state.time - t

            return State(max_speed, state.boost, time, dist)

        return process_boost if cls.use_boost else process_no_boost


class Velocity0To1400(VelocityRange):
    max_speed = THROTTLE_MID_SPEED
    use_boost = False

    @staticmethod
    @jit
    def distance_traveled(t: float, v0: float) -> float:
        b = THROTTLE_ACCELERATION_0
        return (b * (-a * t + math.expm1(a * t)) + a * v0 * math.expm1(a * t)) / (a * a)

    @staticmethod
    @jit
    def velocity_reached(t: float, v0: float) -> float:
        b = THROTTLE_ACCELERATION_0
        return (b * math.expm1(a * t)) / a + v0 * math.exp(a * t)

    @staticmethod
    @jit
    def time_reach_velocity(v: float, v0: float) -> float:
        b = THROTTLE_ACCELERATION_0
        return math.log((a * v + b) / (a * v0 + b)) / a


class Velocity0To1400Boost(VelocityRange):
    max_speed = THROTTLE_MID_SPEED
    use_boost = True

    @staticmethod
    @jit
    def distance_traveled(t: float, v0: float) -> float:
        b = THROTTLE_ACCELERATION_0 + BOOST_ACCELERATION
        return (b * (-a * t + math.expm1(a * t)) + a * v0 * math.expm1(a * t)) / (a * a)

    @staticmethod
    @jit
    def velocity_reached(t: float, v0: float) -> float:
        b = THROTTLE_ACCELERATION_0 + BOOST_ACCELERATION
        return (b * math.expm1(a * t)) / a + v0 * math.exp(a * t)

    @staticmethod
    @jit
    def time_reach_velocity(v: float, v0: float) -> float:
        b = THROTTLE_ACCELERATION_0 + BOOST_ACCELERATION
        return math.log((a * v + b) / (a * v0 + b)) / a


# for when the only acceleration that applies is from boost.
class Velocity1400To2300(Velocity0To1400):
    max_speed = MAX_CAR_SPEED
    use_boost = True

    @staticmethod
    @jit
    def distance_traveled(t: float, v0: float) -> float:
        return t * (BOOST_ACCELERATION * t + 2 * v0) / 2

    @staticmethod
    @jit
    def velocity_reached(t: float, v0: float) -> float:
        return BOOST_ACCELERATION * t + v0

    @staticmethod
    @jit
    def time_reach_velocity(v: float, v0: float) -> float:
        return (v - v0) / BOOST_ACCELERATION


# for when the velocity is opposite the throttle direction,
# only the breaking acceleration applies, boosting has no effect.
# assuming throttle is positive, flip velocity signs if otherwise.
class VelocityNegative(VelocityRange):
    max_speed = 0.
    use_boost = False

    @staticmethod
    @jit
    def distance_traveled(t: float, v0: float) -> float:
        return t * (BREAK_ACCELERATION * t + 2 * v0) / 2

    @staticmethod
    @jit
    def velocity_reached(t: float, v0: float) -> float:
        return BREAK_ACCELERATION * t + v0

    @staticmethod
    @jit
    def time_reach_velocity(v: float, v0: float) -> float:
        return (v - v0) / BREAK_ACCELERATION


step1 = VelocityNegative.wrap()
step2 = Velocity0To1400Boost.wrap()
step3 = Velocity0To1400.wrap()
step4 = Velocity1400To2300.wrap()


@vectorize([float64(float64, float64, float64)], nopython=True)
def distance_traveled_numpy(t: float, v0: float, boost_amount: float) -> float:
    """Returns the max distance driven forward using boost, this allows any starting velocity
    assuming we're not using boost when going backwards and using it otherwise."""
    state = State(v0, boost_amount, t, 0.)

    state = step1(state)
    state = step2(state)
    state = step3(state)
    state = step4(state)

    return state.dist + state.time * state.vel


def main():

    from timeit import timeit
    import numpy as np

    time = np.array([1.821725257142, 2.] * 180)
    initial_velocity = np.array([-2300., -400.] * 180)
    boost_amount = np.array([0., 30.] * 180)

    def test_function():
        return distance_traveled_numpy(time, initial_velocity, boost_amount)

    # print(test_function())

    fps = 120
    n_times = 1000000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == '__main__':
    main()
