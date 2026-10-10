import torch

def gradientClip(paramters, maxL2Norm, eps = 1e-6):
    grads = [p.grad for p in paramters if p.grad is not None]
    all_grads = torch.cat([grad.flatten() for grad in grads])
    l2norm = torch.norm(all_grads, 2)
    if l2norm > maxL2Norm:
        factor = maxL2Norm / (l2norm + eps)
        for grad in grads:
            grad.mul_(factor)