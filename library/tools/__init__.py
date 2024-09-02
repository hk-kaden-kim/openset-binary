from .lossReduction import loss_reducer
from .viz import *

import yaml
import numpy
from tqdm import tqdm

_device = None

def get_device():
    return _device

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

# def print_table(unique_values:np.array, value_counts:np.array, max_columns=10):
#     # Calculate the number of rows needed
#     num_rows = len(unique_values) // max_columns + (len(unique_values) % max_columns > 0)

#     # Create an empty table
#     table = np.zeros((num_rows, max_columns), dtype=int)

#     # Fill in the table with value counts
#     for i, count in enumerate(value_counts):
#         row, col = divmod(i, max_columns)
#         table[row, col] = count

#     # Print the table
#     print(f"Total: {sum(value_counts)}")
#     for i, value in enumerate(unique_values):
#         row, col = divmod(i, max_columns)
#         print(f"{value}: {table[row, col]:<10}", end="")
#         if col == max_columns - 1 or i == len(unique_values) - 1:
#             print()

# def dataset_stats(dataset, batch_size, is_verbose=False):
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     label = []
#     for (x, y) in tqdm(data_loader, miniters=int(len(data_loader)/5), maxinterval=600, disable=not is_verbose):
#         label.extend(y.tolist())
#     stats = np.unique(np.array(label), return_counts=True)
#     print_table(stats[0], stats[1])

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