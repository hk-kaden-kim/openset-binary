import torch
from torch.nn import functional as F
import torch.nn as nn
import torch
from .. import tools

def __openset_ovr(k, sigma, logit_values):
    # print(k, sigma)
    # print()
    # assert False, "FLAG!"

    # Option 1.
    # return F.sigmoid(k*(logit_values + sigma)) * F.sigmoid(k*(-logit_values + sigma))

    # Option 2.
    return torch.where(logit_values > 0 , F.sigmoid(-logit_values + sigma), F.sigmoid(logit_values + sigma))


class __OpenSetOvR(torch.autograd.Function):
    """
    The symetric sigmoid activation function.
    
    Function
    if logit < 0, F.sigmoid(logit + sigma)
    if logit >= 0, F.sigmoid(-1 * logit + sigma) )

    Gradient
    if logit < 0, F.sigmoid(logit + sigma) * (1 - F.sigmoid(logit + sigma))
    if logit >= 0, -1 * F.sigmoid(-1 * logit + simga) * (1 - F.sigmoid(-1 * logit + simga))

    """
    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(logits)
        return torch.where(logits < 0, F.sigmoid(logits + 10), F.sigmoid(-1 * logits + 10))
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (logits,) = ctx.saved_tensors

        grad1 = F.sigmoid(logits + 10) * (1 - F.sigmoid(logits + 10))
        grad2 = -1 * F.sigmoid(-1 * logits + 10) * (1 - F.sigmoid(-1 * logits + 10))

        grad = torch.where(logits < 0, grad1, grad2)
        return grad_output * grad
    

class OpenSetOvR(nn.Module):
    """
    The symetric sigmoid activation function.
    
    Function
    if logit < 0, F.sigmoid(logit + sigma)
    if logit >= 0, F.sigmoid(-1 * logit + sigma) )

    Gradient
    if logit < 0, F.sigmoid(logit + sigma) * (1 - F.sigmoid(logit + sigma))
    if logit >= 0, -1 * F.sigmoid(-1 * logit + simga) * (1 - F.sigmoid(-1 * logit + simga))

    """

    def __init__(self, sigma) -> None:
        super().__init__()
        self.sigma = sigma
    
    def forward(self, logits: torch.Tensor, last_layer_weights, norm=True) -> torch.Tensor:

        # Multiply scale factors to apply the same margin for every decision boundary.
        if norm: 
            scale = torch.sqrt(torch.sum(torch.square(last_layer_weights),dim=1))
            logits = logits / scale
        
        return torch.where(logits < 0, F.sigmoid(logits + self.sigma), F.sigmoid(-1 * logits + self.sigma))


# torch.manual_seed(0)

# osovr_act = losses.OpenSetOvR(sigma=10)

# data = torch.randn(4, dtype=torch.double, requires_grad=True)

# if torch.autograd.gradcheck(osovr_act, data, eps=1e-8, atol=1e-7):
#     print("Success!")
# else:
#     print("Fail!")
