import time
from rlbot.agents.base_agent import SimpleControllerState
from skeleton import SkeletonAgent
from .base_action import BaseAction


class BaseTestAgent(SkeletonAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = self.create_action()
        self.initialized = False

    def create_action(self) -> BaseAction:
        raise NotImplementedError

    def get_controls(self) -> SimpleControllerState:
        self.test_process()
        if self.initialized:
            return self.action.get_controls(self.game_data)
        return SimpleControllerState()

    def retire(self):
        self.matchcomms.outgoing_broadcast.put_nowait("pass")
        while not self.matchcomms.outgoing_broadcast.empty():
            time.sleep(0.01)

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
            self.action = self.create_action()
            self.initialized = False

        if self.action.failed:
            outgoing.put_nowait("fail")
            self.action = self.create_action()
            self.initialized = False
