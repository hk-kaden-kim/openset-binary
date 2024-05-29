import functools
import torch

########################################################################
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

def loss_reducer(func):
    @functools.wraps(func)
    def __loss_reducer__(*args, reduction="none", **kwargs):
        result = func(*args, **kwargs)
        if reduction == "none" or reduction is None:
            return result
        elif reduction == "mean":
            return torch.mean(result)
        elif reduction == "sum":
            return torch.sum(result)

    return __loss_reducer__