""""
This code is from David Ruhe's Clifford Group Equivariant Neural Networks repository:
https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
