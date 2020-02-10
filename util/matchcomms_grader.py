from typing import Optional

from rlbot.matchcomms.client import MatchcommsClient
from rlbot.training.training import Grade, Pass
from rlbottraining.grading.grader import Grader
from rlbottraining.grading.training_tick_packet import TrainingTickPacket


class MatchcommsGrader(Grader):
    matchcomms: MatchcommsClient = None
    initialized = False

    def on_tick(self, tick: TrainingTickPacket) -> Optional[Grade]:
        assert self.matchcomms

        if not self.initialized:
            self.matchcomms.outgoing_broadcast.put_nowait("start")

        if not self.matchcomms.incoming_broadcast.empty():
            print(self.matchcomms.incoming_broadcast.get_nowait())
            return Pass()

        return None
