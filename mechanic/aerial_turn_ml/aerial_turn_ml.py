import torch
from rlbot.agents.base_agent import SimpleControllerState
from rlutilities.simulation import Game

from mechanic.aerial_turn_ml.policy import Policy
from mechanic.aerial_turn_ml.simulation import Simulation
from mechanic.base_mechanic import BaseMechanic


class AerialTurnML(BaseMechanic):
    predicted_o: torch.Tensor = None
    predicted_w: torch.Tensor = None

    def __init__(self):
        super().__init__()
        self.policy = Policy()
        self.simulation = Simulation(self.policy)

    def step(self, info: Game) -> SimpleControllerState:
        o = torch.tensor([[info.my_car.rotation[i, j] for j in range(3)] for i in range(3)])[None, :]
        w = torch.tensor([info.my_car.angular_velocity[i] for i in range(3)])[None, :]
        if self.simulation.o is not None and self.simulation.w is not None:
            print(self.simulation.o - o)
            print(self.simulation.w - w)

        self.simulation.o, self.simulation.w = o, w
        self.simulation.step(info.time_delta)

        rpy = self.policy(o, w)
        self.controls.roll, self.controls.pitch, self.controls.yaw = rpy[0]
        return self.controls
