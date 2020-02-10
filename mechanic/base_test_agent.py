from rlbot.agents.base_agent import SimpleControllerState
from skeleton import SkeletonAgent
from .base_mechanic import BaseMechanic


class BaseTestAgent(SkeletonAgent):
    def __init__(self, name, team, index):
        super(BaseTestAgent, self).__init__(name, team, index)
        self.mechanic = self.create_mechanic()
        self.initialized = False

    def get_controls(self) -> SimpleControllerState:
        self.test_process()
        return self.get_mechanic_controls()

    def create_mechanic(self) -> BaseMechanic:
        raise NotImplementedError

    def get_mechanic_controls(self) -> SimpleControllerState:
        raise NotImplementedError

    def test_process(self):
        if not self.initialized and not self.matchcomms.incoming_broadcast.empty():
            self.matchcomms.incoming_broadcast.get_nowait()
            self.initialized = True

        if self.initialized and self.mechanic.finished:
            self.matchcomms.outgoing_broadcast.put_nowait("pass")
            self.initialized = False
