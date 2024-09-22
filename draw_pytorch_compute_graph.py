from torchviz import make_dot, make_dot_from_trace
import torch
from library import architectures

if __name__ == '__main__':
    net = architectures.LeNet(is_osovr=True)
    x = torch.randn((128,1,28,28))
    dot = make_dot(net(x), params=dict(net.named_parameters()),
                   show_attrs=True, show_saved=True)
    dot.render("LeNet_osovr_fail_nograd", format="png",cleanup=True)