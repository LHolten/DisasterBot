import sys
from pathlib import Path
from typing import Optional
from random import randrange
import numpy as np
from util.linear_algebra import normalize

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
from rlbottraining.exercise_runner import run_module, ReloadPolicy
from rlbottraining.match_configs import make_match_config_with_bots
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import TrainingExercise


class RotationExercise(TrainingExercise):
    def on_briefing(self) -> Optional[Grade]:
        self.grader.matchcomms = self.get_matchcomms()
        return None

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        central_location = np.array([randrange(-3500, 3500), randrange(-4500, 4500), randrange(120, 300)])
        ball_state = BallState(
            Physics(
                location=Vector3(central_location[0], central_location[1], central_location[2]),
                velocity=Vector3(0, 0, 0),
            )
        )

        car_direction = normalize(np.array([randrange(-100, 100), randrange(-100, 100), 0]))
        offset = car_direction * randrange(20, 120)
        car_position = central_location + offset
        car_position[2] = 17.01
        car_state = CarState(
            jumped=False,
            double_jumped=False,
            physics=Physics(
                rotation=Rotator(math.pi / 2, 0, 0),
                location=Vector3(car_position[0], car_position[1], car_position[2]),
                angular_velocity=Vector3(0, 0, 0),
                velocity=Vector3(0, 0, 0),
            ),
        )

        return GameState(ball=ball_state, cars={0: car_state})


# this is for first process imports
current_path = Path(__file__).absolute().parent
sys.path.insert(0, str(current_path.parent.parent))

from util.matchcomms_grader import MatchcommsGrader


def make_default_playlist():
    match_config = make_match_config_with_bots(blue_bots=[current_path / "jumping_shot_agent.cfg"])
    exercise = RotationExercise(name="Jumpshot exercise", grader=MatchcommsGrader(), match_config=match_config)
    return [exercise]


if __name__ == "__main__":
    run_module(Path(__file__).absolute(), reload_policy=ReloadPolicy.NEVER)
