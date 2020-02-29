import sys
from pathlib import Path
from typing import Optional

import math
from rlbot.training.training import Grade
from rlbot.utils.game_state_util import GameState, BallState, GameInfoState, Physics, Rotator, Vector3, CarState
from rlbottraining.exercise_runner import run_module, ReloadPolicy
from rlbottraining.match_configs import make_match_config_with_bots
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import TrainingExercise


class DriveExercise(TrainingExercise):
    def on_briefing(self) -> Optional[Grade]:
        self.grader.matchcomms = self.get_matchcomms()
        return None

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:

        random_position = Vector3(rng.uniform(-3000, 3000), rng.uniform(-4000, 4000), 18)
        random_velocity = Vector3(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000), 0)
        random_rotation = Rotator(0, rng.uniform(-math.pi, math.pi), 0)

        car_physics = Physics(
            location=random_position,
            velocity=random_velocity,
            rotation=random_rotation,
            angular_velocity=Vector3(0, 0, 0),
        )

        boost = rng.uniform(33, 100)

        car_state = CarState(boost_amount=boost, physics=car_physics)

        random_position = Vector3(rng.uniform(-3000, 3000), rng.uniform(-4000, 4000), 93)

        ball_state = BallState(
            physics=Physics(location=random_position, velocity=Vector3(0, 0, 0), angular_velocity=Vector3(0, 0, 0))
        )

        game_info = GameInfoState(game_speed=1)

        return GameState(cars={0: car_state}, ball=ball_state, game_info=game_info)


# this is for first process imports
current_path = Path(__file__).absolute().parent
sys.path.insert(0, str(current_path.parent.parent))

from util.matchcomms_grader import MatchcommsGrader


def make_default_playlist():
    match_config = make_match_config_with_bots(blue_bots=[current_path / "drive_agent.cfg"])
    exercise = DriveExercise(name="Drive to the target", grader=MatchcommsGrader(), match_config=match_config)
    return [exercise]


if __name__ == "__main__":
    run_module(Path(__file__).absolute(), reload_policy=ReloadPolicy.NEVER)
