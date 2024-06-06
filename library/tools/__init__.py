from .lossReduction import loss_reducer
from .viz import *
 
_device = None

def target_encoding(target, num_of_classes, init=0, kn_target=1):

    import torch

    # Encode target values
    enc_target = []
    for t in target:
        enc_t = [init] * num_of_classes
        if t > -1:
            enc_t[t] = kn_target
        enc_target.append(enc_t)
        
    if torch.cuda.is_available():
        return torch.tensor(enc_target).to(torch.float).to(_device)
    else:
        return torch.tensor(enc_target).to(torch.float)
########################################################################
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

def device(x):
    global _device
    if _device is None:
        import torch

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(_device)


def set_device_cpu():
    global _device
    import torch
    _device = torch.device("cpu")


def set_device_gpu(index=0):
    global _device
    import torch
    _device = torch.device(f"cuda:{index}")