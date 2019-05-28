import sys
from pathlib import Path
from typing import Optional

import math
from rlbot.training.training import Grade
from rlbot.utils.game_state_util import GameState, BallState, Physics, Rotator, Vector3, CarState
from rlbottraining import exercise_runner
from rlbottraining.match_configs import make_match_config_with_bots
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import TrainingExercise


class RotationExercise(TrainingExercise):
    def on_briefing(self) -> Optional[Grade]:
        self.grader.matchcomms = self.get_matchcomms()
        return None

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        m = 1
        f = rng.choice([-1, 1])

        car_physics = Physics(velocity=Vector3(0, 0, 0),
                              rotation=Rotator(0, (45 + m * 90) / 180 * math.pi, 0),
                              angular_velocity=Vector3(0, 0, 0),
                              location=Vector3(f * m * 2048, m * -2560, 17.148628))

        ball_physics = Physics(location=Vector3(0, 0, 92.739998),
                               velocity=Vector3(0, 0, 0),
                               angular_velocity=Vector3(0, 0, 0))

        car_state = CarState(jumped=False, double_jumped=False, boost_amount=34, physics=car_physics)

        ball_state = BallState(physics=ball_physics)

        return GameState(ball=ball_state, cars={0: car_state})


if __name__ == '__main__':
    current_path = Path(__file__).absolute().parent
    sys.path.insert(0, str(current_path.parent.parent))  # this is for first process imports

    from common_graders.matchcomms_grader import MatchcommsGrader

    match_config = make_match_config_with_bots(blue_bots=[current_path / 'kickoff_agent.cfg'])
    exercise = RotationExercise(name='kickoff', grader=MatchcommsGrader(), match_config=match_config)

    print(next(exercise_runner.run_playlist([exercise])))
