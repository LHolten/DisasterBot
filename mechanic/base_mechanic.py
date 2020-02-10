from rlbot.agents.base_agent import SimpleControllerState


class BaseMechanic:
    def __init__(self, agent):
        self.controls = SimpleControllerState()
        self.agent = agent
        self.finished = False

    def step(self, *args) -> SimpleControllerState:
        raise NotImplementedError
