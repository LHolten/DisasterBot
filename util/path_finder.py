import heapq
from collections import namedtuple

from util.physics.drive_1d_distance import state_at_distance

import numpy as np

from util.linear_algebra import norm, dot

from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


Node = namedtuple("Node", ["time", "vel", "boost", "i", "prev"])


def find_fastest_path(boost_pads: np.ndarray, start: np.ndarray, target: np.ndarray, vel: np.ndarray, boost: float):
    queue = [PrioritizedItem(0, Node(0, vel, boost, -2, None))]

    # -1 is target, -2 is start
    boost_indices = set(range(boost_pads.shape[0])) | {-1, -2}

    while True:
        state: Node = heapq.heappop(queue).item

        if state.i == -1:
            return state

        if state.i not in boost_indices:
            continue

        boost_indices.remove(state.i)

        location = start
        if state.i != -2:
            location = boost_pads[state.i]["location"]

        for i in boost_indices:
            pad_location = target
            if i != -1:
                pad_location = boost_pads[i]["location"]

            relative_location = pad_location - location
            distance = norm(relative_location)
            node_direction = relative_location / distance
            vel_to_node = dot(state.vel, node_direction)
            delta_time, vel, boost = state_at_distance(distance, vel_to_node, state.boost)
            time = state.time + delta_time

            if i != -1:
                pad_time = 10 if boost_pads[i]["is_full_boost"] else 4

                if boost_pads[i]["is_active"] or boost_pads[i]["timer"] + time >= pad_time:
                    pad_boost = 100 if boost_pads[i]["is_full_boost"] else 12
                    boost = min(boost + pad_boost, 100)

            heapq.heappush(queue, PrioritizedItem(time, Node(time, node_direction * vel, boost, i, state)))


def first_target(boost_pads: np.ndarray, target: np.ndarray, route: Node):
    while route.prev.i != -2:
        route = route.prev

    if route.i == -1:
        return target
    return boost_pads[route.i]["location"]


def main():
    """Testing for errors and performance"""

    from timeit import timeit
    from rlbot.utils.structures.game_data_struct import GameTickPacket
    from skeleton.test.skeleton_agent_test import SkeletonAgentTest, MAX_BOOSTS

    agent = SkeletonAgentTest()
    game_tick_packet = GameTickPacket()
    game_tick_packet.num_boost = MAX_BOOSTS
    agent.initialize_agent()

    boost_pads = agent.game_data.boost_pads

    boost_pads[["is_active", "timer"]] = 0
    boost_pads["is_active"] = True

    my_loc = np.array([150, -3500, 20])
    target_loc = np.array([150, 3500, 20])

    def test_function():
        return first_target(boost_pads, target_loc, find_fastest_path(boost_pads, my_loc, target_loc, 100, 50))

    fps = 120
    n_times = 100
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == "__main__":
    main()
