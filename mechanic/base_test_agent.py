from rlbot.agents.base_agent import SimpleControllerState
from skeleton import SkeletonAgent
from .base_mechanic import BaseMechanic


class BaseTestAgent(SkeletonAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = self.create_mechanic()
        self.initialized = False

    def get_controls(self) -> SimpleControllerState:
        self.test_process()
        if self.initialized:
            return self.get_mechanic_controls()
        return SimpleControllerState()

    def create_mechanic(self) -> BaseMechanic:
        raise NotImplementedError

    def get_mechanic_controls(self) -> SimpleControllerState:
        raise NotImplementedError

    def test_process(self):
        incoming = self.matchcomms.incoming_broadcast
        outgoing = self.matchcomms.outgoing_broadcast

        while not incoming.empty():
            message = incoming.get_nowait()
            if message == "start":
                outgoing.put_nowait("initialized")
                self.initialized = True

        if self.mechanic.finished:
            outgoing.put_nowait("pass")
            self.mechanic = self.create_mechanic()
            self.initialized = False

        if self.mechanic.failed:
            outgoing.put_nowait("fail")
            self.mechanic = self.create_mechanic()
            self.initialized = False
