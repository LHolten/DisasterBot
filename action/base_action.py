from rlbot.agents.base_agent import SimpleControllerState


class BaseAction:
    def __init__(self, agent):
        self.controls = SimpleControllerState()
        self.agent = agent
        self.finished = False
        self.failed = False

    def reset_status(self):
        self.__init__(self.agent)

    def get_controls(self, game_data) -> SimpleControllerState:
        raise NotImplementedError

    def update_status(self, game_data) -> bool:
        raise NotImplementedError
