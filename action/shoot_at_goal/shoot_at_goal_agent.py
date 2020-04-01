from action.base_test_agent import BaseTestAgent
from action.shoot_at_goal import ShootAtGoal


class TestAgent(BaseTestAgent):
    def create_action(self):
        return ShootAtGoal(self, rendering_enabled=True)
