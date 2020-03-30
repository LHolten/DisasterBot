from action.base_test_agent import BaseTestAgent
from action.shadow_ball import ShadowBall


class TestAgent(BaseTestAgent):
    def create_action(self):
        return ShadowBall(self, rendering_enabled=True)
