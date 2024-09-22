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
        print("\n↓↓↓ Activation setup ↓↓↓")
        print(f"{self.__class__.__name__} Activation Loaded!")
        super().__init__()
        self.sigma = sigma
        print(f'Sigma : {self.sigma}')
    
    def forward(self, logits: torch.Tensor, last_layer_weights, norm=True) -> torch.Tensor:

        # Multiply scale factors to apply the same margin for every decision boundary.
        # if norm: 
            # with torch.no_grad():
            #     scale = torch.sqrt(torch.sum(torch.square(last_layer_weights),dim=1))
            # logits = (logits / scale)
        
        return torch.where(logits < 0, F.sigmoid(logits + self.sigma), F.sigmoid(-1 * logits + self.sigma))