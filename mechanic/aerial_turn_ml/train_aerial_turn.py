import sys
from pathlib import Path

import torch
from quicktracer import trace
from torch.optim.adam import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

identity = torch.diag(torch.ones(3))[None, :, :].to(device)


class Trainer:
    def __init__(self):
        from mechanic.aerial_turn_ml.policy import Policy
        from mechanic.aerial_turn_ml.simulation import Simulation

        self.policy = Policy().to(device)
        # self.policy.load_state_dict(torch.load('policy.mdl'))
        self.simulation = Simulation(self.policy)
        self.optimizer = Adam(self.policy.parameters())
        # self.optimizer.load_state_dict(torch.load('optimizer.state'))

    def train(self):
        try:
            while True:
                self.episode()
        except KeyboardInterrupt:
            pass
            torch.save(self.policy.state_dict(), 'policy.mdl')
            torch.save(self.optimizer.state_dict(), 'optimizer.state')

    def episode(self):
        self.simulation.random_state()
        loss = torch.zeros(self.simulation.o.shape[0], device=device)

        for i in range(90):
            self.simulation.step(1 / 30)

            loss += angle_between(self.simulation.o, identity)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        trace(loss.mean().item())


def angle_between(u: torch.Tensor, v: torch.Tensor):
    mask = torch.diag(torch.ones(3)).byte().to(device)
    meps = 1 - 1e-5
    return torch.acos(meps * 0.5 * (torch.sum(torch.sum(u[:, :, None, :] * v[:, None, :, :], 3)[:, mask], 1) - 1.0))


if __name__ == '__main__':
    current_path = Path(__file__).absolute().parent
    sys.path.insert(0, str(current_path.parent.parent))  # this is for first process imports

    trainer = Trainer()
    trainer.train()
