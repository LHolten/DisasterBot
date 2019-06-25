import torch
import sys
from pathlib import Path

if __name__ == "__main__":
    current_path = Path(__file__).absolute().parent
    sys.path.insert(0, str(current_path.parent.parent))  # this is for first process imports

    from mechanic.aerial_turn_ml.policy import Policy

    policy = Policy(20)
    policy.load_state_dict(torch.load("full_rotation_20.mdl"))

    example = torch.rand(1, 3, 3), torch.rand(1, 3)

    torch_script_module = torch.jit.trace(policy, example)

    torch_script_module.save('orientation.pt')
