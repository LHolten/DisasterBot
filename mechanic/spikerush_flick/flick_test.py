import sys
from pathlib import Path
from typing import Optional
import copy

import math
from rlbot.training.training import Grade
from rlbot.utils.game_state_util import GameState, BallState, Physics, Rotator, Vector3, CarState
from rlbottraining import exercise_runner
from rlbottraining.match_configs import make_match_config_with_bots
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import TrainingExercise


class FlickExercise(TrainingExercise):
    def on_briefing(self) -> Optional[Grade]:
        self.grader.matchcomms = self.get_matchcomms()
        return None

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        car_physics = Physics()
        car_physics.rotation = Rotator(0, rng.uniform(-math.pi, math.pi), 0)
        car_physics.location = Vector3(0, 0, 16.5)
        car_physics.velocity = Vector3(rng.uniform(-20, 20), rng.uniform(-20, 20), 0)
        car_physics.angular_velocity = Vector3(0, 0, 0)

        ball_physics = copy.deepcopy(car_physics)
        ball_physics.location.z += 92.75 + 16.5

        return GameState(cars={0: CarState(physics=car_physics)}, ball=BallState(physics=ball_physics))


if __name__ == '__main__':
    current_path = Path(__file__).absolute().parent
    sys.path.insert(0, str(current_path.parent.parent))  # this is for first process imports

    from common_graders.matchcomms_grader import MatchcommsGrader

    match_config = make_match_config_with_bots(blue_bots=[current_path / 'flick_agent.cfg'])
    match_config.mutators.rumble = 'Spike Rush'
    match_config.mutators.respawn_time = '1 Second'

    exercises = [FlickExercise(name='flick', grader=MatchcommsGrader(), match_config=match_config)
                 for _ in range(100)]

    print(list(exercise_runner.run_playlist(exercises)))
