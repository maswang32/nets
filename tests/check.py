import numpy as np
import torch


def check_equals(x, y):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    print(np.max(np.abs(x - y)))
