import math

from policy.base_policy import BasePolicy
from action.base_action import BaseAction

from action.collect_boost import CollectBoost
from action.kickoff import Kickoff


class ExamplePolicy(BasePolicy):

    def __init__(self, agent):
        super(ExamplePolicy, self).__init__(agent)
        self.collect_boost_action = CollectBoost(agent)
        self.kickoff_action = Kickoff(agent)

    def get_default_action(self) -> BaseAction:
        return Kickoff(self.agent)

    def get_action(self, game_data) -> BaseAction:

        self.action.update_status(game_data)

        ball_loc = game_data.ball.location
        kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 140

        interrupt = kickoff

        if not self.action.finished and not interrupt:
            return self.action

        self.action.reset_status()

        if kickoff:
            self.action = self.kickoff_action
        else:
            self.action = self.collect_boost_action

        return self.action
