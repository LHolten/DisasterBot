from pathlib import Path
from typing import Optional

import math
from rlbot.matchcomms.client import MatchcommsClient
from rlbot.training.training import Grade, Pass
from rlbot.utils.game_state_util import GameState, BallState, Physics, Rotator, Vector3, CarState
from rlbottraining import exercise_runner
from rlbottraining.grading.grader import Grader
from rlbottraining.grading.training_tick_packet import TrainingTickPacket
from rlbottraining.match_configs import make_match_config_with_bots
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import TrainingExercise


class MatchcommsGrader(Grader):
    matchcomms: MatchcommsClient = None
    initialized = False

    def on_tick(self, tick: TrainingTickPacket) -> Optional[Grade]:
        assert self.matchcomms

        if not self.initialized:
            self.matchcomms.outgoing_broadcast.put_nowait('start')

        if not self.matchcomms.incoming_broadcast.empty():
            print(self.matchcomms.incoming_broadcast.get_nowait())
            return Pass()

        return None


class RotationExercise(TrainingExercise):
    grader: MatchcommsGrader

    def on_briefing(self) -> Optional[Grade]:
        self.grader.matchcomms = self.get_matchcomms()
        return None

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        car_physics = Physics()
        car_physics.rotation = Rotator(rng.uniform(-math.pi / 2, math.pi / 2),
                                       rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi))
        car_physics.location = Vector3(rng.uniform(-1000, 1000),
                                       rng.uniform(-1000, 1000), rng.uniform(50, 1400))
        car_physics.angular_velocity = Vector3(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5))

        ball_state = BallState(physics=Physics(velocity=Vector3(0, 0, 20), location=Vector3(0, 0, 800)))

        return GameState(cars={0: CarState(physics=car_physics)}, ball=ball_state)


if __name__ == '__main__':
    current_path = Path(__file__).absolute().parent

    match_config = make_match_config_with_bots(blue_bots=[current_path / 'test_agent.cfg'])
    exercise = RotationExercise(name='rotate to target', grader=MatchcommsGrader(), match_config=match_config)

    print(next(exercise_runner.run_playlist([exercise])))
