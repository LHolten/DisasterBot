from action.base_test_agent import BaseTestAgent
from action.hit_ground_ball import HitGroundBall


class TestAgent(BaseTestAgent):
    def create_action(self):
        return HitGroundBall(self, rendering_enabled=True)
