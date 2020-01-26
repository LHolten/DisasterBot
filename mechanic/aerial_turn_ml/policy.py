import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU


class Actor(Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = Linear(12, hidden_size)
        self.linear2 = Linear(hidden_size, 3)
        self.softsign = ReLU()

    def forward(self, o: Tensor, w: Tensor):
        flat_data = torch.cat((o.flatten(1, 2), w), 1)
        return self.linear2(self.softsign(self.linear1(flat_data)))


class Policy(Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size)
        self.symmetry = True

    def forward(self, o: Tensor, w: Tensor):
        if self.symmetry:
            o = o[:, None, :, :].repeat(1, 4, 1, 1)
            w = w[:, None, :].repeat(1, 4, 1)

            o[:, 0:2, :, 0].neg_()
            o[:, ::2, :, 1].neg_()

            o[:, 0:2, 0].neg_()
            o[:, ::2, 1].neg_()

            w[:, 0:2, 1].neg_()
            w[:, ::2, 0].neg_()
            w[:, 0:2, 2].neg_()
            w[:, ::2, 2].neg_()

            rpy: Tensor = self.actor(o.flatten(0, 1), w.flatten(0, 1)).view(-1, 4, 3)

            rpy[:, 0:2, 1].neg_()
            rpy[:, ::2, 0].neg_()
            rpy[:, 0:2, 2].neg_()
            rpy[:, ::2, 2].neg_()

            return torch.clamp(rpy.mean(1), -1, 1)

        else:
            rpy: Tensor = self.actor(o, w)

            return torch.clamp(rpy, -1, 1)
