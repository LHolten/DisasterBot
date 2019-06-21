import torch
from rlbot.agents.base_agent import SimpleControllerState
from rlutilities.simulation import Game

from mechanic.aerial_turn_ml.policy import Policy
from mechanic.aerial_turn_ml.simulation import Simulation
from mechanic.base_mechanic import BaseMechanic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

identity = torch.diag(torch.ones(3))[None, :, :].to(device)
hidden_size = 20
model_name = f'full_rotation_{hidden_size}'


class AerialTurnML(BaseMechanic):
    # predicted_o: torch.Tensor = None
    # predicted_w: torch.Tensor = None
    frames_done = 0

    def __init__(self):
        super().__init__()
        self.policy = Policy(hidden_size).to(device)
        self.policy.load_state_dict(torch.load(model_name + '.mdl'))
        self.simulation = Simulation(self.policy)

    def step(self, info: Game) -> SimpleControllerState:
        o = torch.tensor([[info.my_car.rotation[i, j] for j in range(3)] for i in range(3)])[None, :].to(device)
        w = torch.tensor([info.my_car.angular_velocity[i] for i in range(3)])[None, :].to(device)

        # if self.simulation.o is not None and self.simulation.w is not None:
        #     self.simulation.step(info.time_delta)
        #     print(self.simulation.o - o)
        #     print(self.simulation.w - w)

        self.simulation.o, self.simulation.w = o, w

        rpy = self.policy(self.simulation.o.permute(0, 2, 1), self.simulation.w_local())[0]
        self.controls.roll, self.controls.pitch, self.controls.yaw = rpy

        if self.simulation.error()[0].item() < 0.1:
            self.frames_done += 1
        else:
            self.frames_done = 0

        if self.frames_done >= 1:
            self.finished = True

        return self.controls
