import sys
from pathlib import Path
from rlbot.utils.structures.game_data_struct import (
    FieldInfoPacket,
    GameTickPacket,
    MAX_BOOSTS,
    MAX_GOALS,
    MAX_PLAYERS,
)

from rlbot.utils.structures.ball_prediction_struct import BallPrediction, MAX_SLICES

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from skeleton import SkeletonAgent


class SkeletonAgentTest(SkeletonAgent):

    """A base class for all SkeletonAgent tests"""

    def get_field_info(self):
        field_info = FieldInfoPacket()
        field_info.num_boosts = MAX_BOOSTS
        field_info.num_goals = MAX_GOALS
        return field_info

    def get_ball_prediction_struct(self):
        ball_prediction = BallPrediction()
        ball_prediction.num_slices = MAX_SLICES
        return ball_prediction


def main():
    """Testing for errors and performance"""

    from timeit import timeit

    # import cProfile
    # import pstats
    # import io
    # from pstats import SortKey

    # pr = cProfile.Profile()
    # pr.enable()

    agent = SkeletonAgentTest("test_agent", 0, 0)
    game_tick_packet = GameTickPacket()
    game_tick_packet.num_cars = MAX_PLAYERS
    game_tick_packet.num_boost = MAX_BOOSTS
    game_tick_packet.num_tiles = MAX_GOALS

    agent.initialize_agent()

    def test_function():
        return agent.get_output(game_tick_packet)

    test_function()

    fps = 120
    n_times = 1000
    time_taken = timeit(test_function, number=n_times)
    percentage = time_taken * fps / n_times * 100

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage:.5f} % of our time budget.")

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())


if __name__ == "__main__":
    main()
