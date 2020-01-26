import torch

from torch.optim.optimizer import Optimizer


class Yeet(Optimizer):
    """Implements Yeet algorithm by Hytak.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): rho = 1 gives SGD, rho = 0 is a bit like newtons method (default: 0.5)
        lr (float, optional): the initial learning rate (default: 1e-6)
    """

    def __init__(self, params, lr=1e-6, rho=0.5):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))

        defaults = dict(lr=lr, rho=rho)
        super(Yeet, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if p.grad.is_sparse:
                        raise RuntimeError('Yeet does not support sparse gradients')
                    state = self.state[p]

                    rho, lr = group['rho'], group['lr']

                    # State initialization
                    if len(state) == 0:
                        state['previous_g'] = p.grad.clone()
                        state['previous_p'] = p.clone()

                    delta_g = (p.grad - state['previous_g']).abs()
                    delta_p = (p - state['previous_p']).abs()

                    inv_hessian = delta_p / delta_g
                    inv_hessian = torch.where(torch.isnan(inv_hessian), torch.zeros_like(p), inv_hessian)

                    step = p.grad * (inv_hessian + lr)

                    # Trust region
                    if rho != 0:
                        max_step = delta_p / rho + lr * p.grad.abs()
                        step = torch.where(step < -max_step, -max_step, step)
                        step = torch.where(step > max_step, max_step, step)

                    state['previous_g'] = p.grad.clone()
                    state['previous_p'] = p.clone()

                    p.sub_(step)

        return loss


def andt(*t: torch.Tensor):
    t = torch.stack(t)
    mask = t <= 0
    only_one = mask.sum(0) == 1
    done = mask.sum(0) == 0
    return (t * mask.float()).sum(0) * only_one.float() + done.float()
