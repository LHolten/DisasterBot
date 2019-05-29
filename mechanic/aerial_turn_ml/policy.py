import torch
from torch import Tensor
from torch.nn import Module, Linear, Softsign


class Policy(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(12, 3)
        self.softsign = Softsign()

    def forward(self, o: Tensor, w: Tensor):
        flat_data = torch.cat((w, o.flatten(1, 2)), dim=1)
        return self.softsign(self.linear1(flat_data))
