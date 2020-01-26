import sys
from pathlib import Path
import msvcrt
import math

import torch
from torch.optim.adadelta import Adadelta
from quicktracer import trace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
steps = 40
delta_time = 1 / 30
hidden_size = 40
load = True
rotation_eps = 0.01
model_name = f'full_rotation_{hidden_size}_yeet_0.01'


class Trainer:
    def __init__(self):
        from mechanic.aerial_turn_ml.policy import Policy
        from mechanic.aerial_turn_ml.simulation import Simulation
        from mechanic.aerial_turn_ml.optimizer import Yeet, andt

        self.policy = Policy(hidden_size).to(device)
        self.simulation = Simulation(self.policy)
        self.optimizer = Yeet(self.policy.parameters())
        # self.optimizer = Adadelta(self.policy.parameters())
        self.andt = andt

        self.max_reward = 0

        if load:
            self.policy.load_state_dict(torch.load(model_name + '.mdl'), False)
            self.optimizer.load_state_dict(torch.load(model_name + '.state'))

        for group in self.optimizer.param_groups:
            group['rho'] = 0.5
            group['lr'] = 0.0002

    def train(self):
        while not msvcrt.kbhit():
            self.episode()

        torch.save(self.policy.state_dict(), model_name + '.mdl')
        torch.save(self.optimizer.state_dict(), model_name + '.state')

    def episode(self):
        self.simulation.random_state()
        reward = torch.zeros((self.simulation.o.shape[0], steps), device=device)

        for i in range(steps):
            self.simulation.step(delta_time)

            reward[:, i] = self.simulation.error().neg() + rotation_eps

        trace((reward > 0).float().sum(1).mean(0).item(), reset_on_parent_change=False, key='frames done')

        reward[:, steps - 1] = self.andt(reward[:, steps - 1])
        for i in reversed(range(steps - 1)):
            reward[:, i] = self.andt(reward[:, i], reward[:, i+1])

        loss = reward.sum(1).mean(0).neg()

        # if average_reward.item() > self.max_reward:
        #     self.max_reward = average_reward.item()
        #     torch.save(self.policy.state_dict(), f'{model_name}_{round(self.max_reward, 1)}.mdl')
        #     torch.save(self.optimizer.state_dict(), f'{model_name}_{round(self.max_reward, 1)}.state')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        trace(loss.item(), reset_on_parent_change=False, key='loss')
        trace((reward < 0).sum(1).float().mean(0).item(), reset_on_parent_change=False, key='frame weight')


if __name__ == '__main__':
    current_path = Path(__file__).absolute().parent
    sys.path.insert(0, str(current_path.parent.parent))  # this is for first process imports

    torch.autograd.set_detect_anomaly(True)

    trainer = Trainer()
    trainer.train()
