import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim

from library import architectures, tools, losses, dataset

import pathlib
import random
import numpy as np

import time


from functools import partial

import os
os.environ['RAY_DEDUP_LOGS'] = '0'
           
import ray
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig
from ray.tune import CLIReporter
from ray.tune.experiment.trial import Trial


########################################################################
# Reference Code
# 
# Author: Manuel Günther
# Date: 2024
# Availability: https://gitlab.uzh.ch/manuel.guenther/eos-example
########################################################################

SCALE = 'SmallScale'
APPROACH = 'focal' # wclass        focal
W_CLASS_TYPE = 'None'     # None    global    batch

def set_seeds(seed):
    """ Sets the seed for different sources of randomness.

    Args:
        seed(int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class mbc_loss_config():
    def __init__(self, option, wclass_type='global', wclass_weight_unkn=1, focal_gamma=0, focal_alpha='None', focal_weight_unkn=1):
        self.option = option
        self.schedule = 'None'
        self.wclass_type = wclass_type
        self.wclass_weight_unkn = wclass_weight_unkn
        self.mining_alpha = 1
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.focal_weight_unkn = focal_weight_unkn
        self.focal_mining = 1
        
def train_mbc(config):

    # PARAMETERS
    SCALE = 'SmallScale'
    data_root = '/local/scratch/hkim' if SCALE == 'SmallScale' else '/local/scratch/datasets/ImageNet/ILSVRC2012'
    protocol_root = '../../data/LargeScale'
    arch_name = 'LeNet_plus_plus' if SCALE == 'SmallScale' else 'ResNet_50'
    # gpu = ray.get_gpu_ids()
    lr = 1.e-3
    epochs = 70 if SCALE == 'SmallScale' else 120
    batch_size = 128 if SCALE == 'SmallScale' else 64
    num_workers = 5

    if APPROACH == 'wclass':
        # Class weighting > Unknown Weight Tuning
        loss_config = mbc_loss_config(option='wclass',
                                    wclass_type=W_CLASS_TYPE, wclass_weight_unkn=config['unkn_weight'])
    elif APPROACH == 'focal':
        # Focal Loss > Gamma Tuning
        loss_config = mbc_loss_config(option='focal',
                                      focal_gamma=config['f_gamma'], focal_alpha=W_CLASS_TYPE, focal_weight_unkn=1)
    else:
        loss_config = None
        assert False, f"APPROACH Wrong : {APPROACH}"

    # SETTING. Dataset
    if SCALE == 'SmallScale':
        data = dataset.EMNIST(data_root)
    else:
        data = dataset.IMAGENET(data_root, protocol_root = protocol_root, protocol = int(SCALE.split('_')[1]))
    training_data, validation_data, num_classes = data.get_train_set(include_negatives=True, has_background_class=False)
    
    train_data_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # SETTING. Loss function
    gt_labels = None
    if loss_config.option == 'wclass' and loss_config.wclass_type == 'global':
        gt_labels = dataset.get_gt_labels(training_data, is_verbose=False)
    if loss_config.option == 'focal' and loss_config.focal_alpha == 'global':
        gt_labels = dataset.get_gt_labels(training_data, is_verbose=False)
    loss_func=losses.multi_binary_loss(num_of_classes=num_classes, gt_labels=gt_labels, loss_config=loss_config, epochs=epochs, is_verbose=False)
    
    # SETTING. NN Architecture
    net = architectures.__dict__[arch_name](use_BG=False, num_classes=num_classes, final_layer_bias=True)
    net = tools.device(net)

    # SETTING. Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)

    #######################################################
    # Train
    #######################################################
    for epoch in range(1, epochs + 1, 1):
        t0 = time.time() # Check a duration of one epoch.

        train_accuracy = torch.zeros(2, dtype=int)
        train_confidence = torch.zeros(4, dtype=float)
        net.train()
        num_batch = 0
        for x, y in train_data_loader:

            x = tools.device(x)
            y = tools.device(y)
            optimizer.zero_grad()
            logits, _ = net(x)
            
            loss = loss_func(logits, y, epoch)

            # metrics on training set
            train_accuracy += losses.accuracy(logits, y)
            train_confidence += losses.confidence_v2(logits, y,
                                                    offset = 0.,
                                                    unknown_class = num_classes-1,
                                                    last_valid_class = None,
                                                    is_binary=True)
            loss.backward()
            optimizer.step()
            num_batch += 1
        # train_confidence = torch.mul(train_confidence, torch.Tensor([1/num_batch, 1, 1/num_batch, 1]))

        # Validation
        with torch.no_grad():
            val_accuracy = torch.zeros(2, dtype=int)
            val_confidence = torch.zeros(4, dtype=float)
            net.eval()

            num_batch = 0
            for x,y in val_data_loader:
                # predict
                x = tools.device(x)
                y = tools.device(y)
                logits, _ = net(x)
                
                loss = loss_func(logits, y, epoch)

                # metrics on validation set
                val_accuracy += losses.accuracy(logits, y)
                val_confidence += losses.confidence_v2(logits, y,
                                                        offset = 0.,
                                                        unknown_class = -1,
                                                        last_valid_class = None,
                                                        is_binary=True)
                num_batch += 1
        
            # val_confidence = torch.mul(val_confidence, torch.Tensor([1/num_batch, 1, 1/num_batch, 1]))
        
        # Report out
        t_kn_conf = float(train_confidence[0] / train_confidence[1])
        t_un_conf = float(train_confidence[2] / train_confidence[3])
        v_kn_conf = float(val_confidence[0] / val_confidence[1])
        v_un_conf = float(val_confidence[2] / val_confidence[3])
        curr_score = float(val_confidence[0] / val_confidence[1]) + float(val_confidence[2] / val_confidence[3])

        train.report({'train_conf':((t_kn_conf+t_un_conf)/2),
                     'val_conf':((v_kn_conf+v_un_conf)/2),
                     'model_score':(curr_score)})
        
    print("Finished Training")

class TrialTerminationReporter(CLIReporter):
    def __init__(self):
        super(TrialTerminationReporter, self).__init__()
        self.num_terminated = 0

    def should_report(self, trials, done=False):
        """Reports only on trial termination events."""
        old_num_terminated = self.num_terminated
        self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
        return self.num_terminated > old_num_terminated


def main():

    ray.init(configure_logging=False, _temp_dir='/home/user/hkim/UZH-MT/openset-binary/_ray')
    seed = 42
    set_seeds(seed)

    if APPROACH == 'wclass':
        # Class weighting > Unknown Weight Tuning
        param_space={
            "unkn_weight": tune.grid_search([0.1, 0.5, 0.7, 1, 5, 7, 10, 30, 50, 70]),
        }
    elif APPROACH == 'focal':
        # Focal Loss > Gamma Tuning 
        param_space={
            "f_gamma": tune.grid_search([1, 2, 3, 4, 5]),
        }
    else:
        param_space = None
        assert False, f"APPROACH Wrong : {APPROACH}"

    tuner = tune.Tuner(
        tune.with_resources(
            train_mbc,
            resources={"gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                grace_period=10 if SCALE=='SmallScale' else 20,
            ),
            metric="model_score",
            mode="max",
            num_samples=1,
        ),
        run_config=RunConfig(
            progress_reporter=TrialTerminationReporter(),
            storage_path = '/home/user/hkim/UZH-MT/openset-binary/_ray'
        ),
        param_space=param_space,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="model_score", mode="max")

    print(f"Best trial config: {best_result.config}\n\n")
    print(f"{SCALE} {APPROACH} {W_CLASS_TYPE}")
    print("Hyperparameter Tuning Done!")

if __name__ == "__main__":
    print(f"{SCALE} {APPROACH} {W_CLASS_TYPE}")
    main()