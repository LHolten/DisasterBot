import heapq
from collections import namedtuple

from rlbot.utils.structures.game_data_struct import FieldInfoPacket, MAX_BOOSTS

from util.drive_physics_simulation import (
    min_travel_time_simulation,
    boost_reached_simulation,
    velocity_reached_simulation,
)

import numpy as np

from util.linear_algebra import norm

Node = namedtuple("Node", ["time", "vel", "boost", "i", "prev"])


def find_fastest_path(boost_pads: np.ndarray, start: np.ndarray, target: np.ndarray, vel: float, boost: float):
    queue = [Node(0, vel, boost, -2, None)]

    # -1 is target, -2 is start
    boost_indices = set(range(boost_pads.shape[0])) | {-1, -2}

    while True:
        state: Node = heapq.heappop(queue)

        if state.i == -1:
            return state

        if state.i not in boost_indices:
            continue

        boost_indices.remove(state.i)

        location = start
        if state.i != -2:
            location = boost_pads[state.i]["location"]

        for i in boost_indices:
            pad_boost = 0
            pad_location = target
            if i != -1:
                pad_boost = 100 if boost_pads[i]["is_full_boost"] else 12
                pad_location = boost_pads[i]["location"]

            distance = norm(location - pad_location)
            delta_time = min_travel_time_simulation(distance, state.vel, state.boost)
            vel = velocity_reached_simulation(delta_time, state.vel, state.boost)
            boost = min(boost_reached_simulation(delta_time, state.vel, state.boost) + pad_boost, 100)
            heapq.heappush(queue, Node(state.time + delta_time, vel, boost, i, state))


def first_target(boost_pads: np.ndarray, target: np.ndarray, route: Node):
    while route.prev.i != -2:
        route = route.prev

    if route.i == -1:
        return target
    return boost_pads[route.i]["location"]


def main():
    """Testing for errors and performance"""

    from timeit import timeit
    import ctypes

    my_loc = np.array([150, -3500, 20])
    target_loc = np.array([150, 3500, 20])

    field_info = FieldInfoPacket()
    boost_pads = field_info.boost_pads
    num_boosts = MAX_BOOSTS

    buf_from_mem = ctypes.pythonapi.PyMemoryView_FromMemory
    buf_from_mem.restype = ctypes.py_object
    buf_from_mem.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)

    dtype = np.dtype([("location", "<f4", 3), ("is_full_boost", "?")], True)
    buffer = buf_from_mem(ctypes.addressof(boost_pads), dtype.itemsize * num_boosts, 0x100)
    converted_boost_pads = np.frombuffer(buffer, dtype)

    full_dtype = [
        ("location", "<f4", 3),
        ("is_full_boost", "?"),
        ("is_active", "?"),
        ("timer", "<f4"),
    ]

    boost_pads = np.zeros(num_boosts, full_dtype)
    boost_pads[["location", "is_full_boost"]] = converted_boost_pads

    boost_pads[["is_active", "timer"]] = 0
    boost_pads["is_active"] = True

    def test_function():
        return first_target(boost_pads, target_loc, find_fastest_path(boost_pads, my_loc, target_loc, 100, 50))

    fps = 120
    n_times = 10
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == "__main__":
    main()
