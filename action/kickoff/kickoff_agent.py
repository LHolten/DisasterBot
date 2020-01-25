from action.base_test_agent import BaseTestAgent
from action.kickoff.kickoff import Kickoff


class TestAgent(BaseTestAgent):
    def create_action(self):
        return Kickoff(self)
