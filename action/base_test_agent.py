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
        if self.initialized:
            return self.action.get_controls(self.game_data)
        return SimpleControllerState()

    def test_process(self):
        incoming = self.matchcomms.incoming_broadcast
        outgoing = self.matchcomms.outgoing_broadcast

        while not incoming.empty():
            message = incoming.get_nowait()
            if message == "start":
                outgoing.put_nowait("initialized")
                self.initialized = True

        if self.action.finished:
            outgoing.put_nowait("pass")

        if self.action.failed:
            outgoing.put_nowait("fail")
