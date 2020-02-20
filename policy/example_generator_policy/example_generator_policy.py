import math

from policy.base_policy import BasePolicy
from action.base_action import BaseAction

from action.collect_boost import CollectBoost
from action.kickoff import Kickoff
from action.hit_ground_ball import HitGroundBall

from typing import Generator


class ExamplePolicy(BasePolicy):
    def __init__(self, agent, rendering_enabled=True):
        super(ExamplePolicy, self).__init__(agent, rendering_enabled)
        self.collect_boost_action = CollectBoost(agent, rendering_enabled)
        self.kickoff_action = Kickoff(agent, rendering_enabled)
        self.hit_ball_action = HitGroundBall(agent, rendering_enabled)
        self.action = None

    def get_action(self, game_data) -> BaseAction:
        ball_loc = game_data.ball.location
        kickoff = math.sqrt(ball_loc[0] ** 2 + ball_loc[1] ** 2) < 1

        if kickoff:
            # reset the action loop
            self.action = None
            return self.kickoff_action
        else:
            if self.action is None:
                self.action = self.action_loop(game_data)
            return self.action.send(game_data)

    def action_loop(self, game_data) -> Generator[BaseAction]:
        while True:
            # choose action to do
            if game_data.my_car.boost > 20:
                action = self.collect_boost_action
            else:
                action = self.collect_boost_action

            # use action until it is finished
            while not action.finished:
                game_data = yield action

            action.reset_status()
