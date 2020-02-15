import math
from collections import namedtuple

from numba import vectorize, guvectorize, f8, jit
import numpy as np

from util.special_lambertw import lambertw

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
b2 = THROTTLE_ACCELERATION_0 + BOOST_ACCELERATION

fast_jit = jit(f8(f8, f8), nopython=True, fastmath=True)

State = namedtuple("State", ["dist", "vel", "boost", "time"])


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

    @staticmethod
    def time_travel_distance(d: float, v0: float) -> float:
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
                if cls_max_speed <= state.vel or state.time == 0.0 or state.boost <= 0:
                    return state

                time_0_boost = state.boost / BOOST_CONSUMPTION_RATE
                time_vel = cls_time_reach_velocity(cls_max_speed, state.vel)

                if state.time <= time_0_boost and state.time <= time_vel:
                    dist = state.dist + cls_distance_traveled(state.time, state.vel)
                    vel = cls_velocity_reached(state.time, state.vel)
                    boost = state.boost - state.time * BOOST_CONSUMPTION_RATE
                    return State(dist, vel, boost, 0.0)

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

                return State(dist, vel, boost, time)

        else:

            def distance_state_step(state: State) -> State:
                if cls_max_speed <= state.vel or state.time == 0.0:
                    return state

                time_vel = cls_time_reach_velocity(cls_max_speed, state.vel)

                if state.time <= time_vel:
                    dist = state.dist + cls_distance_traveled(state.time, state.vel)
                    vel = cls_velocity_reached(state.time, state.vel)
                    return State(dist, vel, state.boost, 0.0)

                dist = state.dist + cls_distance_traveled(time_vel, state.vel)
                time = state.time - time_vel

                return State(dist, cls_max_speed, state.boost, time)

        return jit(distance_state_step, nopython=True, fastmath=True)

    @classmethod
    def wrap_time_reach_velocity_step(cls):
        """Advances the state to the soonest phase end."""

        cls_max_speed = cls.max_speed
        cls_velocity_reached = cls.velocity_reached
        cls_time_reach_velocity = cls.time_reach_velocity

        if cls.use_boost:

            def time_reach_velocity_step(state: State) -> State:
                if cls_max_speed <= state.vel or state[0] <= state.vel or state.boost <= 0:
                    return state

                time_0_boost = state.boost / BOOST_CONSUMPTION_RATE
                time_vel = cls_time_reach_velocity(cls_max_speed, state.vel)
                vel_0_boost = cls_velocity_reached(time_0_boost, state.vel)

                if state[0] <= vel_0_boost and state[0] <= cls_max_speed:
                    time = state.time + cls_time_reach_velocity(state[0], state.vel)
                    return State(state[0], state[0], state.boost, time)

                if time_0_boost < time_vel:
                    time = state.time + time_0_boost
                    velocity = vel_0_boost
                    boost = 0
                else:
                    time = state.time + time_vel
                    velocity = cls_max_speed
                    boost = state.boost - time_vel * BOOST_CONSUMPTION_RATE

                return State(state[0], velocity, boost, time)

        else:

            def time_reach_velocity_step(state: State) -> State:
                rel_sign = 1 if state.vel < state[0] else -1
                if cls_max_speed <= rel_sign * state.vel or state[0] == state.vel:
                    return state

                if rel_sign * state[0] <= cls_max_speed:
                    time = state.time + cls_time_reach_velocity(state[0] * rel_sign, state.vel * rel_sign)
                    return State(state[0], state[0], state.boost, time)

                time = state.time + cls_time_reach_velocity(cls_max_speed * rel_sign, state.vel * rel_sign)
                velocity = cls_max_speed * rel_sign

                return State(state[0], velocity, state.boost, time)

        return jit(time_reach_velocity_step, nopython=True, fastmath=True)


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

    @staticmethod
    @fast_jit
    def time_travel_distance(d: float, v: float) -> float:
        return (-d * a * a - b * lambertw(-((b + a * v) * math.exp(-(a * (v + a * d)) / b - 1)) / b) - a * v - b) / (
            a * b
        )


class Velocity0To1400Boost(VelocityRange):
    max_speed = THROTTLE_MID_SPEED
    use_boost = True

    @staticmethod
    @fast_jit
    def distance_traveled(t: float, v0: float) -> float:
        return (b2 * (-a * t + math.expm1(a * t)) + a * v0 * math.expm1(a * t)) / (a * a)

    @staticmethod
    @fast_jit
    def velocity_reached(t: float, v0: float) -> float:
        return (b2 * math.expm1(a * t)) / a + v0 * math.exp(a * t)

    @staticmethod
    @fast_jit
    def time_reach_velocity(v: float, v0: float) -> float:
        return math.log((a * v + b2) / (a * v0 + b2)) / a

    @staticmethod
    @fast_jit
    def time_travel_distance(d: float, v: float) -> float:
        return (
            -d * a * a - b2 * lambertw(-((b2 + a * v) * math.exp(-(a * (v + a * d)) / b - 1)) / b2) - a * v - b2
        ) / (a * b2)


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

    @staticmethod
    @fast_jit
    def time_travel_distance(d: float, v: float) -> float:
        return (-v + math.sqrt(2 * BOOST_ACCELERATION * d + math.pow(v, 2))) / BOOST_ACCELERATION


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

    @staticmethod
    @fast_jit
    def time_travel_distance(d: float, v: float) -> float:
        return (-v + math.sqrt(2 * BREAK_ACCELERATION * d + math.pow(v, 2))) / BREAK_ACCELERATION


distance_step_velocity_negative = VelocityNegative.wrap_distance_state_step()
distance_step_velocity_0_1400_boost = Velocity0To1400Boost.wrap_distance_state_step()
distance_step_velocity_0_1400 = Velocity0To1400.wrap_distance_state_step()
distance_step_velocity_1400_2300 = Velocity1400To2300.wrap_distance_state_step()


@jit(f8(f8, f8, f8), nopython=True, fastmath=True)
def distance_traveled(time: float, initial_velocity: float, boost_amount: float) -> float:
    """Returns the max distance driven forward using boost, this allows any starting velocity
    assuming we're not using boost when going backwards and using it otherwise."""
    state = State(0.0, initial_velocity, boost_amount, time)

    state = distance_step_velocity_negative(state)
    state = distance_step_velocity_0_1400_boost(state)
    state = distance_step_velocity_0_1400(state)
    state = distance_step_velocity_1400_2300(state)

    return state.dist + state.time * state.vel


distance_traveled_vectorized = vectorize([f8(f8, f8, f8)], nopython=True)(distance_traveled)

time_step_velocity_negative = VelocityNegative.wrap_time_reach_velocity_step()
time_step_velocity_0_1400_boost = Velocity0To1400Boost.wrap_time_reach_velocity_step()
time_step_velocity_0_1400 = Velocity0To1400.wrap_time_reach_velocity_step()
time_step_velocity_1400_2300 = Velocity1400To2300.wrap_time_reach_velocity_step()


@jit(f8(f8, f8, f8), nopython=True, fastmath=True)
def time_reach_velocity(desired_velocity: float, initial_velocity: float, boost_amount: float) -> float:
    """Returns the time it takes to reach any desired velocity including those that require going backwards."""
    state = State(desired_velocity, initial_velocity, boost_amount, 0.0)

    state = time_step_velocity_negative(state)
    state = time_step_velocity_0_1400_boost(state)
    state = time_step_velocity_0_1400(state)
    state = time_step_velocity_1400_2300(state)

    if state[0] != state.vel:
        return 10.0
    return state.time


time_reach_velocity_vectorized = vectorize([f8(f8, f8, f8)], nopython=True)(time_reach_velocity)


def main():

    from timeit import timeit

    time = np.linspace(0, 6, 360)
    initial_velocity = np.linspace(-2300, 2300, 360)
    desired_vel = -initial_velocity
    boost_amount = np.linspace(0, 100, 360)

    def test_function():
        # return distance_traveled_vectorized(time, initial_velocity, boost_amount)
        return time_reach_velocity_vectorized(desired_vel, initial_velocity, boost_amount)

    print(test_function())

    fps = 120
    n_times = 10000
    time_taken = timeit(test_function, number=n_times)
    percentage = time_taken * fps / n_times * 100

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage:.5f} % of our time budget.")


if __name__ == "__main__":
    main()

    # print("Starting distance_traveled accuracy test...")

    # from drive_physics_experimental import distance_traveled as distance_traveled2

    # for time in range(0, 100):
    #     time = time / 10
    #     for initial_velocity in range(0, 2300, 100):
    #         for boost_amount in range(0, 100, 10):
    #             res1 = distance_traveled(time, initial_velocity, boost_amount)
    #             res2 = distance_traveled2(time, initial_velocity, boost_amount)
    #             if abs(res1 - res2) > 1e-4:
    #                 print("Failed the accuracy test")
    #                 print(time, initial_velocity, boost_amount, " : ", res1, res2)
    #                 quit()

    # print("No inaccuracies found.")

    # print("Starting time_reach_velocity accuracy test...")

    # from drive_physics_experimental import time_reach_velocity as time_reach_velocity2

    # for desired_vel in range(-2300, 2300, 100):
    #     for initial_velocity in range(-2300, 2300, 100):
    #         for boost_amount in range(0, 100, 10):
    #             res1 = time_reach_velocity(desired_vel, initial_velocity, boost_amount)
    #             res2 = time_reach_velocity2(desired_vel, initial_velocity, boost_amount)
    #             if abs(res1 - res2) > 1e-4:
    #                 print("Failed the accuracy test")
    #                 print(desired_vel, initial_velocity, boost_amount, " : ", res1, res2)
    #                 quit()

    # print("No inaccuracies found.")
