import math
from collections import namedtuple
from numba import jit, f8

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

    @classmethod
    def wrap_time_travel_distance_step(cls):
        """Advances the state to the soonest phase end."""

        cls_max_speed = cls.max_speed
        cls_distance_traveled = cls.distance_traveled
        cls_velocity_reached = cls.velocity_reached
        cls_time_reach_velocity = cls.time_reach_velocity
        cls_time_travel_distance = cls.time_travel_distance

        if cls.use_boost:

            def time_travel_distance_state_step(state: State) -> State:
                if cls_max_speed <= state.vel or state.dist == 0.0 or state.boost <= 0:
                    return state

                time_0_boost = state.boost / BOOST_CONSUMPTION_RATE
                time_vel = cls_time_reach_velocity(cls_max_speed, state.vel)

                if time_0_boost <= time_vel:
                    dist_0_boost = cls_distance_traveled(time_0_boost, state.vel)
                    if state.dist >= dist_0_boost:
                        time = state.time + time_0_boost
                        vel = cls_velocity_reached(time_0_boost, state.vel)
                        dist = state.dist - dist_0_boost
                        return State(dist, vel, 0.0, time)
                else:
                    dist_vel = cls_distance_traveled(time_vel, state.vel)
                    if state.dist >= dist_vel:
                        time = state.time + time_vel
                        dist = state.dist - dist_vel
                        boost = state.boost - time_vel * BOOST_CONSUMPTION_RATE
                        return State(dist, cls_max_speed, boost, time)

                delta_time = cls_time_travel_distance(state.dist, state.vel)
                time = state.time + delta_time
                vel = cls_velocity_reached(delta_time, state.vel)
                boost = state.boost - delta_time * BOOST_CONSUMPTION_RATE

                return State(0.0, vel, boost, time)

        else:

            def time_travel_distance_state_step(state: State) -> State:
                if cls_max_speed <= state.vel or state.dist == 0.0:
                    return state

                time_vel = cls_time_reach_velocity(cls_max_speed, state.vel)
                dist_vel = cls_distance_traveled(time_vel, state.vel)

                if state.dist <= dist_vel:
                    delta_time = cls_time_travel_distance(state.dist, state.vel)
                    time = state.time + delta_time
                    vel = cls_velocity_reached(delta_time, state.vel)
                    return State(0.0, vel, state.boost, time)
                else:
                    time = state.time + time_vel
                    vel = cls_max_speed
                    dist = state.dist - dist_vel
                    return State(dist, vel, state.boost, time)

        return jit(time_travel_distance_state_step, nopython=True, fastmath=True)


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
            -d * a * a - b2 * lambertw(-((b2 + a * v) * math.exp(-(a * (v + a * d)) / b2 - 1)) / b2) - a * v - b2
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
