import math

DT = 1 / 60

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


def sign(x):
    return 1 if x >= 0 else -1


def throttle_acc(vel, throttle=1):
    throttle = min(1, max(-1, throttle))
    if throttle * vel < 0:
        return -3600 * sign(vel)
    elif throttle == 0:
        return -525 * sign(vel)
    else:
        return (a * min(abs(vel), THROTTLE_MID_SPEED) + b) * throttle


def min_travel_time_simulation(max_distance, v_0, initial_boost):

    time = 0
    distance = 0
    velocity = v_0
    boost = initial_boost
    acceleration = 0

    while (distance < max_distance):
        acceleration = throttle_acc(velocity) + (boost > 1) * BOOST_ACCEL
        velocity = min(velocity + acceleration * DT, MAX_CAR_SPEED)
        distance = distance + velocity * DT + 0.5 * acceleration * DT * DT
        boost -= BOOST_CONSUMPTION_RATE * DT
        time += DT
        if (time > 10):
            break

    return time


def distance_traveled_0_1400(t, v0, is_boosting):
    b2 = b + is_boosting * BOOST_ACCEL
    return (b2 * (-a * t + math.expm1(a * t)) + a * v0 * math.expm1(a * t)) / math.pow(a, 2)


def velocity_reached_0_1400(t, v0, is_boosting):
    b2 = b + is_boosting * BOOST_ACCEL
    return (b2 * math.expm1(a * t)) / a + v0 * math.exp(a * t)


def time_to_reach_velocity_0_1400(v, v0, is_boosting):
    b2 = b + is_boosting * BOOST_ACCEL
    return math.log((a * v + b2) / (a * v0 + b2)) / a


# for when the only acceleration that applies is from boost.

def distance_traveled_1400_2300_boost(t, v0):
    return t * (BOOST_ACCEL * t + 2 * v0) / 2


def velocity_reached_1400_2300_boost(t, v0):
    return BOOST_ACCEL * t + v0


def time_to_reach_velocity_1400_2300_boost(v, v0):
    return (v - v0) / BOOST_ACCEL


def time_to_reach_distance_1400_2300_boost(d, v):
    return -(math.sqrt(2 * BOOST_ACCEL * d + math.pow(v, 2)) + v) / a


# for when the velocity is opposite the throttle direction,
# only the breaking acceleration applies, boosting has no effect.
# assuming throttle is positive, flip velocity signs if otherwise.

def distance_traveled_negative(t, v0):
    return t * (BREAK_ACCEL * t + 2 * v0) / 2


def velocity_reached_negative(t, v0):
    return BREAK_ACCEL * t + v0


def time_to_reach_velocity_negative(v, v0):
    return (v - v0) / BREAK_ACCEL


def distance_traveled_zero_acceleration(t, v0):
    return t * v0


# boost consumption
def time_to_reach_0_boost(boost_amount):
    return boost_amount / BOOST_CONSUMPTION_RATE


def boost_reached(time, initial_boost_amount):
    return max(0, initial_boost_amount - time * BOOST_CONSUMPTION_RATE)


# this allows any starting velocity
# assuming we're throttling forward and using boost efficiently
def distance_traveled(t, v0, boost_amount):

    distance = 0
    time_left = t
    velocity = v0
    boost_left = boost_amount

    if velocity < 0:

        time_0_vel = time_to_reach_velocity_negative(0, velocity)
        if time_left <= time_0_vel:
            return distance + distance_traveled_negative(time_left, velocity)
        else:
            distance = distance + distance_traveled_negative(time_0_vel, velocity)
            time_left = time_left - time_0_vel
            velocity = 0

    if 0 <= velocity < THROTTLE_MID_SPEED:

        if boost_left > 0:
            time_0_boost = time_to_reach_0_boost(boost_left)
            time_1400_vel = time_to_reach_velocity_0_1400(THROTTLE_MID_SPEED, velocity, True)

            if time_0_boost <= time_1400_vel:
                if time_left <= time_0_boost:
                    return distance + distance_traveled_0_1400(time_left, velocity, True)
                else:
                    distance = distance + distance_traveled_0_1400(time_0_boost, velocity, True)
                    velocity = velocity_reached_0_1400(time_0_boost, velocity, True)
                    time_left = time_left - time_0_boost
                    boost_left = 0
            else:
                if time_left <= time_1400_vel:
                    return distance + distance_traveled_0_1400(time_left, velocity, True)
                else:
                    distance = distance + distance_traveled_0_1400(time_1400_vel, velocity, True)
                    boost_left = boost_reached(time_1400_vel, boost_left)
                    time_left = time_left - time_1400_vel
                    velocity = THROTTLE_MID_SPEED

        if boost_left <= 0:
            time_1400_vel = time_to_reach_velocity_0_1400(THROTTLE_MID_SPEED, velocity, False)
            if time_left <= time_1400_vel:
                return distance + distance_traveled_0_1400(time_left, velocity, False)
            else:
                distance = distance + distance_traveled_0_1400(time_1400_vel, velocity, False)
                boost_left = boost_reached(time_1400_vel, boost_left)
                time_left = time_left - time_1400_vel
                velocity = THROTTLE_MID_SPEED

    if THROTTLE_MID_SPEED <= velocity < MAX_CAR_SPEED:

        if boost_left > 0:
            time_0_boost = time_to_reach_0_boost(boost_left)
            time_2300_vel = time_to_reach_velocity_1400_2300_boost(MAX_CAR_SPEED, velocity)

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

    distance = distance + distance_traveled_zero_acceleration(time_left, velocity)

    return distance


def main():

    from timeit import timeit

    time = 1
    initial_velocity = -400
    boost_amount = 30

    def test_function():
        return distance_traveled(time, initial_velocity, boost_amount)

    print(test_function())

    fps = 120
    n_times = 10000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == '__main__':
    main()
