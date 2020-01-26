import torch
from torch.onnx.utils import export
import sys
from pathlib import Path
import numpy

if __name__ == "__main__":
    current_path = Path(__file__).absolute().parent
    sys.path.insert(0, str(current_path.parent.parent))  # this is for first process imports

    from mechanic.aerial_turn_ml.policy import Policy

    policy = Policy(20)
    policy.load_state_dict(torch.load("full_rotation_20.mdl"))

    # print(policy.actor.linear1.weight[3, 2])
    # print(policy.actor.linear1.bias[4])
    # print(policy.actor.linear2.weight[2, 6])
    #
    # print(torch.mm(policy.actor.linear1.weight, torch.ones(12, 1)))
    print(policy(torch.ones(1,3,3), torch.ones(1,3)))

    # example = torch.rand(1, 3, 3), torch.rand(1, 3)

    # torch_script_module = torch.jit.trace(policy, example)
    #
    # torch_script_module.save('orientation.pt')

    # export(policy, example, "full_rotation_20.onnx", verbose=True)

    # data = [
    #     policy.actor.linear1.weight.detach().numpy().flatten('F'),
    #     policy.actor.linear1.bias.detach().numpy().flatten('F'),
    #     policy.actor.linear2.weight.detach().numpy().flatten('F'),
    # ]
    #
    # data = numpy.concatenate(data)
    #
    # numpy.memmap('orientation.bin', dtype=numpy.float32, mode='w+', shape=data.shape)[:] = data[:]
    # print(data.shape)
