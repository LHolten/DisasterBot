import sys
from pathlib import Path
from typing import Optional

import math

from rlbot.training.training import Grade
from rlbot.utils.game_state_util import GameState, BallState, Physics, Rotator, Vector3, CarState
from rlbottraining.exercise_runner import run_module, ReloadPolicy
from rlbottraining.match_configs import make_match_config_with_bots
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import TrainingExercise


class RotationExercise(TrainingExercise):
    def on_briefing(self) -> Optional[Grade]:
        self.grader.matchcomms = self.get_matchcomms()
        return None

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        t = rng.choice([-1, 1])
        f = rng.choice([-1, 1])

        car_physics = Physics(
            velocity=Vector3(0, 0, 0),
            rotation=Rotator(0, (45 * f + t * 90) / 180 * math.pi, 0),
            angular_velocity=Vector3(0, 0, 0),
            location=Vector3(f * t * 2048, t * -2560, 17.0),
        )

        ball_physics = Physics(
            location=Vector3(0, 0, 92.74), velocity=Vector3(0, 0, 0), angular_velocity=Vector3(0, 0, 0),
        )

        car_state = CarState(boost_amount=34, physics=car_physics)

        ball_state = BallState(physics=ball_physics)

        return GameState(ball=ball_state, cars={0: car_state})


# this is for first process imports
current_path = Path(__file__).absolute().parent
sys.path.insert(0, str(current_path.parent.parent))

from util.matchcomms_grader import MatchcommsGrader


def make_default_playlist():
    match_config = make_match_config_with_bots(blue_bots=[current_path / "kickoff_agent.cfg"])
    exercise = RotationExercise(name="Kickoff", grader=MatchcommsGrader(), match_config=match_config)
    return [exercise]


if __name__ == "__main__":
    run_module(Path(__file__).absolute(), reload_policy=ReloadPolicy.NEVER)
