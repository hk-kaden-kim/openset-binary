########################################################################
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

from .lossReduction import loss_reducer

_device = None

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

