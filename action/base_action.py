from rlbot.agents.base_agent import SimpleControllerState
from rlutilities.simulation import Game


class BaseAction:

    def __init__(self, renderer):
        self.controls = SimpleControllerState()
        self.renderer = renderer
        self.finished = False
        self.failed = False

    def get_output(self, info: Game) -> SimpleControllerState:
        raise NotImplementedError

    def get_possible(self, info: Game) -> bool:
        raise NotImplementedError

    def update_status(self, info: Game) -> bool:
        raise NotImplementedError

    def reset_status(self):
        self.finished = False
        self.failed = False
