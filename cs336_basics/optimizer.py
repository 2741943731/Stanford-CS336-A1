import torch
import math
from torch import nn

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                # m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # p.data.addcmul_(-group['lr'], p.data, group['weight_decay'])

                state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias1 = 1 - beta1 ** state['step']
                bias2 = 1 - beta2 ** state['step']
                # adjusted_lr = group['lr'] * math.sqrt(bias2) / bias1

                step_size = group['lr'] * math.sqrt(bias2) / bias1
                demon = state["v"].sqrt().add_(group["eps"])

                p.data.addcdiv_(state["m"], demon, value=-step_size)
                p.data.add_(p.data, alpha=-group['lr']*group["weight_decay"])
        return loss






