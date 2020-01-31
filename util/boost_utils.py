import numpy as np

from rlbot.utils.structures.game_data_struct import FieldInfoPacket, GameTickPacket, MAX_BOOSTS


def closest_available_boost(my_loc: np.ndarray, boost_pads: np.ndarray) -> np.ndarray:
    """Returns the closest available boost pad to my_loc"""

    distances = np.linalg.norm(boost_pads['location'] - my_loc[None, :], axis=1)
    recharge_time = np.where(boost_pads['is_full_boost'], 10, 4)
    available = boost_pads['is_active'] | (distances / 2300 > recharge_time - boost_pads['timer'])

    available_distances = distances[available]
    if len(available_distances) > 0:
        available_boost = boost_pads[available]
        closest_available_index = np.argmin(available_distances)
        return available_boost[closest_available_index]
    else:
        return None


def main():
    """Testing for errors and performance"""

    from timeit import timeit
    import ctypes

    num_boosts = 50
    my_loc = np.array([150, -3500, 20])

    field_info = FieldInfoPacket()
    boost_pads = field_info.boost_pads
    num_boosts = MAX_BOOSTS

    buf_from_mem = ctypes.pythonapi.PyMemoryView_FromMemory
    buf_from_mem.restype = ctypes.py_object
    buf_from_mem.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)

    dtype = np.dtype([('location', '<f4', 3), ('is_full_boost', '?')], True)
    buffer = buf_from_mem(ctypes.addressof(boost_pads), dtype.itemsize * num_boosts, 0x100)
    converted_boost_pads = np.frombuffer(buffer, dtype)

    full_dtype = [('location', '<f4', 3), ('is_full_boost', '?'),
                  ('is_active', '?'), ('timer', '<f4')]

    boost_pads = np.zeros(num_boosts, full_dtype)
    boost_pads[['location', 'is_full_boost']] = converted_boost_pads

    game_tick_packet = GameTickPacket()

    dtype = np.dtype([('is_active', '?'), ('timer', '<f4')], True)
    converted_game_boosts = np.array(game_tick_packet.game_boosts,
                                     copy=False).view(dtype)[:num_boosts]
    boost_pads[['is_active', 'timer']] = converted_game_boosts
    boost_pads['is_active'] = True

    def test_function():
        return closest_available_boost(my_loc, boost_pads)

    fps = 120
    n_times = 10000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == "__main__":
    main()
