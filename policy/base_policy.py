from action.base_action import BaseAction


class BasePolicy:
    def __init__(self, agent, rendering_enabled=False):
        self.agent = agent
        self.rendering_enabled = rendering_enabled
        self.action: BaseAction = self.get_default_action()

    def get_controls(self, game_data):
        self.action = self.get_action(game_data)
        return self.action.get_controls(game_data)

    def get_default_action(self) -> BaseAction:
        raise NotImplementedError

    def get_action(self, game_data) -> BaseAction:
        raise NotImplementedError
