import sys
from pathlib import Path

import torch
from quicktracer import trace
from torch.optim.adadelta import Adadelta

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):
        from mechanic.aerial_turn_ml.policy import Policy
        from mechanic.aerial_turn_ml.simulation import Simulation

        self.policy = Policy().to(device)
        # self.policy.load_state_dict(torch.load('policy.mdl'))
        self.simulation = Simulation(self.policy)
        self.optimizer = Adadelta(self.policy.parameters())
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
        loss = torch.zeros((self.simulation.o.shape[0], 90), device=device)

        for i in range(90):
            self.simulation.step(1 / 30)

            loss[:, i] = self.simulation.error()

        for i in range(1, 90):
            loss[:, 89 - i] = torch.max(loss[:, 89 - i:90 - i], dim=1)[0]

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        trace(loss.mean().item())


if __name__ == '__main__':
    current_path = Path(__file__).absolute().parent
    sys.path.insert(0, str(current_path.parent.parent))  # this is for first process imports

    torch.autograd.set_detect_anomaly(True)

    trainer = Trainer()
    trainer.train()
