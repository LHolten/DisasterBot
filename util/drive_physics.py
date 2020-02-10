import math
from collections import namedtuple

from numba import vectorize, f8, jit

THROTTLE_ACCELERATION_0 = 1600.0
THROTTLE_ACCELERATION_1400 = 160.0
THROTTLE_MID_SPEED = 1400.0

BOOST_ACCELERATION = 991.6667
BREAK_ACCELERATION = 3500.0

MAX_CAR_SPEED = 2300.0

BOOST_CONSUMPTION_RATE = 33.3  # per second

# constants of the acceleration between 0 to 1400 velocity: acceleration = a * velocity + b
a = -(THROTTLE_ACCELERATION_0 - THROTTLE_ACCELERATION_1400) / THROTTLE_MID_SPEED
b = THROTTLE_ACCELERATION_0

fast_jit = jit(f8(f8, f8), nopython=True, fastmath=True)


State = namedtuple("State", ["vel", "boost", "time", "dist"])


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
    def wrap_distance_state_step(cls):
        """Advances the state to the soonest phase end."""

        cls_max_speed = cls.max_speed
        cls_distance_traveled = cls.distance_traveled
        cls_velocity_reached = cls.velocity_reached
        cls_time_reach_velocity = cls.time_reach_velocity

        if cls.use_boost:

            def distance_state_step(state: State) -> State:
                if cls_max_speed <= state.vel or state.time == 0.0 or state.boost == 0.0:
                    return state

                time_0_boost = state.boost / BOOST_CONSUMPTION_RATE
                time_vel = cls_time_reach_velocity(cls_max_speed, state.vel)

                if state.time <= time_0_boost and state.time <= time_vel:
                    dist = state.dist + cls_distance_traveled(state.time, state.vel)
                    return State(0.0, 0.0, 0.0, dist)

                if time_0_boost < time_vel:
                    delta_time = time_0_boost
                    vel = cls_velocity_reached(time_0_boost, state.vel)
                    boost = 0.0
                else:
                    delta_time = time_vel
                    vel = cls_max_speed
                    boost = state.boost - delta_time * BOOST_CONSUMPTION_RATE

                dist = state.dist + cls_distance_traveled(delta_time, state.vel)
                time = state.time - delta_time

                return State(vel, boost, time, dist)

        else:

            def distance_state_step(state: State) -> State:
                if cls_max_speed <= state.vel or state.time == 0.0:
                    return state

                time_vel = cls_time_reach_velocity(cls_max_speed, state.vel)

                if state.time <= time_vel:
                    dist = state.dist + cls_distance_traveled(state.time, state.vel)
                    return State(0.0, 0.0, 0.0, dist)

                dist = state.dist + cls_distance_traveled(time_vel, state.vel)
                time = state.time - time_vel

                return State(cls_max_speed, state.boost, time, dist)

        return jit(distance_state_step, nopython=True, fastmath=True)


class Velocity0To1400(VelocityRange):
    max_speed = THROTTLE_MID_SPEED
    use_boost = False

    @staticmethod
    @fast_jit
    def distance_traveled(t: float, v0: float) -> float:
        return (b * (-a * t + math.expm1(a * t)) + a * v0 * math.expm1(a * t)) / (a * a)

    @staticmethod
    @fast_jit
    def velocity_reached(t: float, v0: float) -> float:
        return (b * math.expm1(a * t)) / a + v0 * math.exp(a * t)

    @staticmethod
    @fast_jit
    def time_reach_velocity(v: float, v0: float) -> float:
        return math.log((a * v + b) / (a * v0 + b)) / a


class Velocity0To1400Boost(VelocityRange):
    max_speed = THROTTLE_MID_SPEED
    use_boost = True

    @staticmethod
    @fast_jit
    def distance_traveled(t: float, v0: float) -> float:
        b = THROTTLE_ACCELERATION_0 + BOOST_ACCELERATION
        return (b * (-a * t + math.expm1(a * t)) + a * v0 * math.expm1(a * t)) / (a * a)

    @staticmethod
    @fast_jit
    def velocity_reached(t: float, v0: float) -> float:
        b = THROTTLE_ACCELERATION_0 + BOOST_ACCELERATION
        return (b * math.expm1(a * t)) / a + v0 * math.exp(a * t)

    @staticmethod
    @fast_jit
    def time_reach_velocity(v: float, v0: float) -> float:
        b = THROTTLE_ACCELERATION_0 + BOOST_ACCELERATION
        return math.log((a * v + b) / (a * v0 + b)) / a


class Velocity1400To2300(Velocity0To1400):
    """for when the only acceleration that applies is from boost."""

    max_speed = MAX_CAR_SPEED
    use_boost = True

    @staticmethod
    @fast_jit
    def distance_traveled(t: float, v0: float) -> float:
        return t * (BOOST_ACCELERATION * t + 2 * v0) / 2

    @staticmethod
    @fast_jit
    def velocity_reached(t: float, v0: float) -> float:
        return BOOST_ACCELERATION * t + v0

    @staticmethod
    @fast_jit
    def time_reach_velocity(v: float, v0: float) -> float:
        return (v - v0) / BOOST_ACCELERATION


class VelocityNegative(VelocityRange):
    """for when the velocity is opposite the throttle direction,
    only the breaking acceleration applies, boosting has no effect.
    assuming throttle is positive, flip velocity signs if otherwise."""

    max_speed = 0.0
    use_boost = False

    @staticmethod
    @fast_jit
    def distance_traveled(t: float, v0: float) -> float:
        return t * (BREAK_ACCELERATION * t + 2 * v0) / 2

    @staticmethod
    @fast_jit
    def velocity_reached(t: float, v0: float) -> float:
        return BREAK_ACCELERATION * t + v0

    @staticmethod
    @fast_jit
    def time_reach_velocity(v: float, v0: float) -> float:
        return (v - v0) / BREAK_ACCELERATION


distance_step_velocity_negative = VelocityNegative.wrap_distance_state_step()
distance_step_velocity_0_1400_boost = Velocity0To1400Boost.wrap_distance_state_step()
distance_step_velocity_0_1400 = Velocity0To1400.wrap_distance_state_step()
distance_step_velocity_1400_2300 = Velocity1400To2300.wrap_distance_state_step()


@jit(f8(f8, f8, f8), nopython=True, fastmath=True)
def distance_traveled(time: float, initial_velocity: float, boost_amount: float) -> float:
    """Returns the max distance driven forward using boost, this allows any starting velocity
    assuming we're not using boost when going backwards and using it otherwise."""
    state = State(initial_velocity, boost_amount, time, 0.0)

    state = distance_step_velocity_negative(state)
    state = distance_step_velocity_0_1400_boost(state)
    state = distance_step_velocity_0_1400(state)
    state = distance_step_velocity_1400_2300(state)

    return state.dist + state.time * state.vel


distance_traveled_vectorized = vectorize([f8(f8, f8, f8)], nopython=True)(distance_traveled)


def main():

    from timeit import timeit
    import numpy as np

    time = np.linspace(0, 6, 360)
    initial_velocity = np.linspace(-2300, 2300, 360)
    boost_amount = np.linspace(0, 100, 360)

    def test_function():
        return distance_traveled_vectorized(time, initial_velocity, boost_amount)

    print(test_function())

    fps = 120
    n_times = 10000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


def consistency():
    from util.drive_physics_experimental import distance_traveled as distance_traveled2

    for time in range(0, 100):
        time = time / 10
        for initial_velocity in range(0, 2300, 100):
            for boost_amount in range(0, 100, 10):
                res1 = distance_traveled(time, initial_velocity, boost_amount)
                res2 = distance_traveled2(time, initial_velocity, boost_amount)
                if abs(res1 - res2) > 1e-4:
                    print("Failed the accuracy test")
                    print(time, initial_velocity, boost_amount, " : ", res1, res2)
                    quit()


if __name__ == "__main__":
    main()
    consistency()
