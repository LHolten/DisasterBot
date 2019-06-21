import sys
from pathlib import Path

import torch
from quicktracer import trace
from torch.optim.adadelta import Adadelta
# from torch.optim.adamax import Adamax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
steps = 60
delta_time = 1 / 30
hidden_size = 20
model_name = f'full_rotation_{hidden_size}'
load = True
rotation_eps = 0.1


class Trainer:
    def __init__(self):
        from mechanic.aerial_turn_ml.policy import Policy
        from mechanic.aerial_turn_ml.simulation import Simulation

        self.policy = Policy(hidden_size).to(device)
        self.simulation = Simulation(self.policy)
        self.optimizer = Adadelta(self.policy.parameters())
        if load:
            self.policy.load_state_dict(torch.load(model_name + '.mdl'), False)
            self.optimizer.load_state_dict(torch.load(model_name + '.state'))

    def train(self):
        try:
            while True:
                self.episode()
        except KeyboardInterrupt:
            pass
            torch.save(self.policy.state_dict(), model_name + '.mdl')
            torch.save(self.optimizer.state_dict(), model_name + '.state')

    def episode(self):
        self.simulation.random_state()
        loss = torch.zeros((self.simulation.o.shape[0], steps), device=device)

        for i in range(steps):
            self.simulation.step(delta_time)

            loss[:, i] = self.simulation.error()

        reward = torch.exp(-0.5 * (loss / rotation_eps).pow(2)).clone()

        for i in reversed(range(steps - 1)):
            reward[:, i] = torch.min(reward[:, i:i+2], dim=1)[0]

        self.optimizer.zero_grad()
        reward.sum(1).mean(0).neg().backward()
        self.optimizer.step()
        trace(loss.sum(1).mean(0).item())
        trace((loss < rotation_eps).float().sum().item())


if __name__ == '__main__':
    current_path = Path(__file__).absolute().parent
    sys.path.insert(0, str(current_path.parent.parent))  # this is for first process imports

    torch.autograd.set_detect_anomaly(True)

    trainer = Trainer()
    trainer.train()
