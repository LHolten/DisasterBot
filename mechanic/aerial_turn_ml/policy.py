import torch
from torch import Tensor
from torch.nn import Module, Linear, Softsign


class Actor(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(12, 10)
        self.linear2 = Linear(10, 3)
        self.softsign = Softsign()

    def forward(self, o: Tensor, w: Tensor):
        flat_data = torch.cat((o.flatten(1, 2), w), 1)
        return self.linear2(self.softsign(self.linear1(flat_data)))


class Policy(Module):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.softsign = Softsign()
        self.symmetry = True

    def forward(self, o: Tensor, w: Tensor):
        if self.symmetry:
            o = o[:, None, :, :].repeat(1, 8, 1, 1)
            w = w[:, None, :].repeat(1, 8, 1)

            o[:, 0:4, :, 0] = o[:, 0:4, :, 0].neg()
            o[:, 0:2, :, 1] = o[:, 0:2, :, 1].neg()
            o[:, 4:6, :, 1] = o[:, 4:6, :, 1].neg()
            o[:, ::2, :, 2] = o[:, ::2, :, 2].neg()

            o[:, 0:4, 0] = o[:, 0:4, 0].neg()
            o[:, 0:2, 1] = o[:, 0:2, 1].neg()
            o[:, 4:6, 1] = o[:, 4:6, 1].neg()
            o[:, ::2, 2] = o[:, ::2, 2].neg()

            w[:, 0:4, [1, 2]] = w[:, 0:4, [1, 2]].neg()
            w[:, 0:2, [0, 2]] = w[:, 0:2, [0, 2]].neg()
            w[:, 4:6, [0, 2]] = w[:, 4:6, [0, 2]].neg()
            w[:, ::2, [0, 1]] = w[:, ::2, [0, 1]].neg()

            rpy: Tensor = self.actor(o.flatten(0, 1), w.flatten(0, 1)).view(-1, 8, 3)

            rpy[:, 0:4, [1, 2]] = rpy[:, 0:4, [1, 2]].neg()
            rpy[:, 0:2, [0, 2]] = rpy[:, 0:2, [0, 2]].neg()
            rpy[:, 4:6, [0, 2]] = rpy[:, 4:6, [0, 2]].neg()
            rpy[:, ::2, [0, 1]] = rpy[:, ::2, [0, 1]].neg()

            return self.softsign(rpy[:, 7])

        else:
            rpy: Tensor = self.actor(o, w)

            return self.softsign(rpy)
