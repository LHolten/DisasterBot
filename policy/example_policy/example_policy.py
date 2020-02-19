import math

from policy.base_policy import BasePolicy
from action.base_action import BaseAction

from action.collect_boost import CollectBoost
from action.kickoff import Kickoff
from action.hit_ground_ball import HitGroundBall


class ExamplePolicy(BasePolicy):
    def __init__(self, agent, rendering_enabled=True):
        super(ExamplePolicy, self).__init__(agent, rendering_enabled)
        self.collect_boost_action = CollectBoost(agent, rendering_enabled)
        self.kickoff_action = Kickoff(agent, rendering_enabled)
        self.hit_ball_action = HitGroundBall(agent, rendering_enabled)

    def get_default_action(self) -> BaseAction:
        return Kickoff(self.agent, self.rendering_enabled)

    def get_action(self, game_data) -> BaseAction:

        ball_loc = game_data.ball.location
        kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 1

        interrupt = kickoff

        # keep doing the same action until finished or interrupted
        if not self.action.finished and not interrupt:
            return self.action

        self.action.reset_status()

        # simple decision tree to choose the next action
        if kickoff:
            self.action = self.kickoff_action
        else:
            if game_data.my_car.boost > 20:
                self.action = self.hit_ball_action
            else:
                self.action = self.collect_boost_action

        return self.action
