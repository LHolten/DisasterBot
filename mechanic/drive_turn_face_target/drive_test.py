import sys
from pathlib import Path
from typing import Optional

import math
from rlbot.training.training import Grade
from rlbot.utils.game_state_util import (
    GameState,
    BallState,
    Physics,
    Rotator,
    Vector3,
    CarState,
    GameInfoState,
)
from rlbottraining import exercise_runner
from rlbottraining.match_configs import make_match_config_with_bots
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import TrainingExercise


class RotationExercise(TrainingExercise):
    def on_briefing(self) -> Optional[Grade]:
        self.grader.matchcomms = self.get_matchcomms()
        return None

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        random_position = Vector3(rng.uniform(-2000, 1000), rng.uniform(-2000, 2000), 18)
        random_velocity = Vector3(rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), 0)
        random_rotation = Rotator(0, rng.uniform(-math.pi, math.pi), 0)
        random_ang_vel = Vector3(0, 0, rng.uniform(-5.5, 5.5))

        car_physics = Physics(
            location=random_position,
            velocity=random_velocity,
            rotation=random_rotation,
            angular_velocity=random_ang_vel,
        )

        boost = rng.uniform(0, 50)

        car_state = CarState(boost_amount=boost, physics=car_physics)

        random_position = Vector3(rng.uniform(-2000, 2000), rng.uniform(-2000, 2000), 93)

        ball_state = BallState(
            physics=Physics(location=random_position, velocity=Vector3(0, 0, 0), angular_velocity=Vector3(0, 0, 0),)
        )

        game_info = GameInfoState(game_speed=0.25)

        return GameState(ball=ball_state, cars={0: car_state}, game_info=game_info)


# this is for first process imports
current_path = Path(__file__).absolute().parent
sys.path.insert(0, str(current_path.parent.parent))

from util.matchcomms_grader import MatchcommsGrader


def make_default_playlist():
    match_config = make_match_config_with_bots(blue_bots=[current_path / "drive_agent.cfg"])
    exercise = RotationExercise(name="Face the ball", grader=MatchcommsGrader(), match_config=match_config)
    return [exercise]


if __name__ == "__main__":
    exercise_runner.run_module(Path(__file__).absolute())
