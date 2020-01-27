import sys
from pathlib import Path
from timeit import timeit

from rlbot.utils.structures.game_data_struct import FieldInfoPacket, GameTickPacket
from rlbot.utils.structures.ball_prediction_struct import BallPrediction

current_path = Path(__file__).absolute().parent
print(current_path.parent.parent)
sys.path.insert(0, str(current_path.parent.parent))

from skeleton import SkeletonAgent


class SkeletonAgentTest(SkeletonAgent):

    """A base class for all SkeletonAgent tests"""

    def get_field_info(self):
        return FieldInfoPacket()

    def get_ball_prediction_struct(self):
        ball_prediction = BallPrediction()
        ball_prediction.num_slices = 360
        return ball_prediction


def main():
    """Testing for errors and performance"""

    agent = SkeletonAgentTest("test_agent", 0, 0)
    game_tick_packet = GameTickPacket()

    def test_function():
        return agent.get_output(game_tick_packet)

    fps = 120
    n_times = 1000
    time_taken = timeit(test_function, number=n_times)
    percentage = round(time_taken * fps / n_times * 100, 5)

    print(f"Took {time_taken} seconds to run {n_times} times.")
    print(f"That's {percentage} % of our time budget.")


if __name__ == "__main__":
    main()
