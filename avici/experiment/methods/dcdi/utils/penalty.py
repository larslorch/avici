import torch

def compute_penalty(list_, p=2, target=0.):
    penalty = 0
    for m in list_:
        penalty += torch.norm(m - target, p=p) ** p
    return penalty
