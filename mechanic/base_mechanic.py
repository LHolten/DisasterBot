from rlbot.agents.base_agent import SimpleControllerState


class BaseMechanic:
    def __init__(self, agent, rendering_enabled=False):
        self.controls = SimpleControllerState()
        self.agent = agent
        self.rendering_enabled = rendering_enabled
        self.finished = False

    def step(self, *args) -> SimpleControllerState:
        raise NotImplementedError
