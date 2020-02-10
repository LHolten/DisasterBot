from rlbot.agents.base_agent import SimpleControllerState
from skeleton import SkeletonAgent
from .base_action import BaseAction


class BaseTestAgent(SkeletonAgent):
    def __init__(self, name, team, index):
        super(BaseTestAgent, self).__init__(name, team, index)
        self.action = self.create_action()
        self.initialized = False

    def create_action(self) -> BaseAction:
        raise NotImplementedError

    def get_controls(self) -> SimpleControllerState:
        self.test_process()
        self.action.update_status(self.game_data)
        return self.action.get_controls(self.game_data)

    def test_process(self):
        if not self.initialized and not self.matchcomms.incoming_broadcast.empty():
            self.matchcomms.incoming_broadcast.get_nowait()
            self.initialized = True

        if self.initialized:
            self.action.update_status(self.game_data)

        if self.initialized and self.action.finished:
            self.matchcomms.outgoing_broadcast.put_nowait("done")
            self.initialized = False
