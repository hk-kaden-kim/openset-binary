from .lossReduction import loss_reducer
from .. import dataset
from .. import losses

import yaml
import numpy
import random
import argparse

from torch import nn
import torch

###################################
# for Interface
###################################
class NameSpace:
    def __init__(self, config):
        # recurse through config
        config = {name : NameSpace(value) if isinstance(value, dict) else value for name, value in config.items()}
        self.__dict__.update(config)

    def __repr__(self):
        return "\n".join(k+": " + str(v) for k,v in vars(self).items())

    def dump(self, indent=4):
        return yaml.dump(self.dict(), indent=indent)

    def dict(self):
        return {k: v.dict() if isinstance(v, NameSpace) else v for k,v in vars(self).items()}

def load_yaml(yaml_file):
    """Loads a YAML file into a nested namespace object"""
    config = yaml.safe_load(open(yaml_file, 'r'))
    return NameSpace(config)

def print_table(unique_values:numpy.array, value_counts:numpy.array, max_columns=10):
    # Calculate the number of rows needed
    num_rows = len(unique_values) // max_columns + (len(unique_values) % max_columns > 0)

    # Create an empty table
    table = numpy.zeros((num_rows, max_columns), dtype=int)

    # Fill in the table with value counts
    for i, count in enumerate(value_counts):
        row, col = divmod(i, max_columns)
        table[row, col] = count

    # Print the table
    print(f"Total: {sum(value_counts)}")
    for i, value in enumerate(unique_values):
        row, col = divmod(i, max_columns)
        print(f"{value}: {table[row, col]:<10}", end="")
        if col == max_columns - 1 or i == len(unique_values) - 1:
            print()
    print()

def train_command_line_options():


    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", "-cf", default='config/train.yaml', help="The configuration file that defines the experiment")
    parser.add_argument("--scale", "-sc", required=True, choices=['SmallScale', 'LargeScale', 'LargeScale_1', 'LargeScale_2', 'LargeScale_3'], help="Choose the scale of training dataset.")
    parser.add_argument("--arch", "-ar", required=True)
    parser.add_argument("--approach", "-ap", required=True, choices=['SoftMax', 'EOS', 'OvR', 'OpenSetOvR', 'etc'])
    parser.add_argument("--seed", "-s", default=42, nargs="+", type=int)
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()

def eval_command_line_options():

    labels={
    "SoftMax" : "Plain SoftMax",
    "EOS" : "Entropic Open-Set",
    "OvR" : "One-vs-Rest Classifiers",
    "etc": None
    }

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", "-cf", default='config/eval.yaml', help="The configuration file that defines the experiment")
    parser.add_argument("--scale", "-sc", required=True, choices=['SmallScale', 'LargeScale', 'LargeScale_1', 'LargeScale_2', 'LargeScale_3'], help="Choose the scale of evaluation dataset.")
    parser.add_argument("--arch", "-ar", required=True)
    parser.add_argument("--approach", "-ap", nargs="+", default=list(labels.keys()), choices=list(labels.keys()), help = "Select the approaches to evaluate; non-existing models will automatically be skipped")
    parser.add_argument("--seed", "-s", default=42, nargs="+", type=int)
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()

###################################
# for Models
###################################
def get_data_and_loss(args, config, arch_name, seed):
    """...TBD..."""

    if args.scale == 'SmallScale':
        data = dataset.EMNIST(config.data.smallscale.root, 
                              split_ratio = config.data.smallscale.split_ratio, 
                              seed = seed)
    else:
        data = dataset.IMAGENET(config.data.largescale.root, 
                                protocol_root = config.data.largescale.protocol, 
                                protocol = int(args.scale.split('_')[1]),
                                is_verbose=True)
    
    if args.approach == "SoftMax":
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, size_train_negatives=config.data.train_neg_size)
        loss_func=nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    
    elif args.approach == "EOS":
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, size_train_negatives=config.data.train_neg_size)
        loss_func=losses.entropic_openset_loss(num_of_classes=num_classes, unkn_weight=config.loss.eos.unkn_weight)

    elif args.approach == 'OvR':
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, size_train_negatives=config.data.train_neg_size)
        loss_func=losses.OvRLoss(num_of_classes=num_classes, mode=config.loss.ovr.mode, training_data=training_data)

    return dict(
                loss_func=loss_func,
                training_data = training_data,
                val_data = val_data,
                num_classes = num_classes
            )

def target_encoding(target, num_of_classes, init=0, kn_target=1):

    import torch

    # Encode target values
    enc_target = []
    for t in target:
        enc_t = [init] * num_of_classes
        if t > -1:
            enc_t[int(t)] = kn_target
        enc_target.append(enc_t)
        
    if torch.cuda.is_available():
        return torch.tensor(enc_target).to(torch.float).to(_device)
    else:
        return torch.tensor(enc_target).to(torch.float)

def check_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().tolist())
    return ave_grads, layers

def check_fc2_weights(net,):
    return net.fc2.weight.mean(dim=0).tolist()

def set_seeds(seed):
    """ Sets the seed for different sources of randomness.

    Args:
        seed(int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

    print(f"Seed: {seed}")


###################################
# for Environment
###################################
_device = None

def device(x):
    global _device
    if _device is None:
        import torch
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # _device = torch.device(get_device() if torch.cuda.is_available() else "cpu")
    return x.to(_device)

def set_device_cpu():
    global _device
    import torch
    _device = torch.device("cpu")

def set_device_gpu(index=0):
    global _device
    import torch
    _device = torch.device(f"cuda:{index}")

def get_device():
    return _device
