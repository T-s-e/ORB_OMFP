"""Double-precision array operations."""

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tt(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float64, device=DEVICE)


def cpu(value):
    return value.detach().cpu().numpy()
