from skeleton import SkeletonAgent
from policy.example_policy.example_policy import ExamplePolicy


class DisasterBot(SkeletonAgent):

    def __init__(self, name, team, index):
        super(DisasterBot, self).__init__(name, team, index)
        self.policy = ExamplePolicy(self)

    def get_controls(self):

        action = self.policy.get_action(self.game_data)
        controls = action.get_controls(self.game_data)

        return controls
