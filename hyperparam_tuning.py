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
import math

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

import logging

SCALE = 'LargeScale_2' # SmallScale LargeScale_2
APPROACH = 'OpenSetOvR'
ARCH = 'ResNet_50'        # LeNet_plus_plus         ResNet_50
# --------------------------------
MODE = None
SIGMA = [5,6,8,11,15]     # 5, 6, 7, 8, 9, 10
MODE_PARAM = [None]
GPUs = "3,4,6,7"
# --------------------------------
# MODE = "F"   
# MODE_PARAM = [0.2,0.4,0.6,0.8,1,2]
# GPUs = "0,1,3,4,5,6,7"
# --------------------------------
# MODE = "C"
# MODE_PARAM = ["global","batch"] 
# GPUs = "3,5"
# --------------------------------
# MODE = "M"
# MODE_PARAM = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
# GPUs = "0,1,3,4,5,6,7"
# --------------------------------

os.environ["CUDA_VISIBLE_DEVICES"]=GPUs

def set_seeds(seed):
    """ Sets the seed for different sources of randomness.

    Args:
        seed(int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def train_mbc(config):

    # PARAMETERS
    data_root = '/local/scratch/hkim' if SCALE == 'SmallScale' else '/local/scratch/datasets/ImageNet/ILSVRC2012'
    protocol_root = '/home/user/hkim/UZH-MT/openset-binary/data/LargeScale'
    arch_name = ARCH
    lr = 1.e-3
    epochs = 70 if SCALE == 'SmallScale' else 120
    batch_size = 128 if SCALE == 'SmallScale' else 64
    num_workers = 5

    if APPROACH == 'OpenSetOvR':
        if MODE == None:
            loss_config = tools.NameSpace({"sigma":config['sigma'], "mode":None})
        else:
            loss_config = tools.NameSpace({"sigma":config['sigma'], "mode":{MODE:config['mode_param']}})
    else:
        loss_config = None
        assert False, f"APPROACH Wrong : {APPROACH}"

    # SETTING. Dataset
    if SCALE == 'SmallScale':
        data = dataset.EMNIST(data_root)
    else:
        data = dataset.IMAGENET(data_root, protocol_root = protocol_root, protocol = int(SCALE.split('_')[1]), is_verbose=False)
    training_data, validation_data, num_classes = data.get_train_set(size_train_negatives=-1 if MODE else 0, has_background_class=False)
    
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
    if APPROACH == 'OpenSetOvR':
        loss_func=losses.OSOvRLoss(num_of_classes=num_classes, loss_config=loss_config, training_data=training_data, is_verbose=False)
    else:
        loss_func=nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        
    # SETTING. NN Architecture
    net = architectures.__dict__[arch_name](use_BG=False, num_classes=num_classes, is_verbose=False)
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

        for x, y in train_data_loader:
            x = tools.device(x)
            y = tools.device(y)
            optimizer.zero_grad()
            logits, _ = net(x)

            # calculate losses
            if APPROACH == 'OpenSetOvR':
                loss = loss_func(logits, y, last_layer_weights = net.fc2.weight.data)
            else:
                loss = loss_func(logits, y)

            # metrics on training set
            if APPROACH == 'OpenSetOvR':
                scores = loss_func.osovr_act(logits, net.fc2.weight.data)
                train_confidence += losses.confidence(scores, y,
                                                        offset = 0.,
                                                        unknown_class = -1,
                                                        last_valid_class = None,)
            else:
                scores = torch.nn.functional.softmax(logits, dim=1)
                train_confidence += losses.confidence(scores, y,
                                                        offset = 1. / num_classes,
                                                        unknown_class = -1,
                                                        last_valid_class = None,)
            train_accuracy += losses.accuracy(scores, y)

            # update the network weights
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            val_accuracy = torch.zeros(2, dtype=int)
            val_confidence = torch.zeros(4, dtype=float)
            net.eval()

            for x,y in val_data_loader:
                # predict
                x = tools.device(x)
                y = tools.device(y)
                logits, _ = net(x)
                
                # loss calculation
                if APPROACH == 'OpenSetOvR':
                    loss = loss_func(logits, y, last_layer_weights = net.fc2.weight.data)
                else:
                    loss = loss_func(logits, y)

                # metrics on validation set
                if APPROACH == 'OpenSetOvR':
                    scores = loss_func.osovr_act(logits, net.fc2.weight.data)
                    val_confidence += losses.confidence(scores, y,
                                                            offset = 0.,
                                                            unknown_class = -1,
                                                            last_valid_class = None,)
                else:
                    scores = torch.nn.functional.softmax(logits, dim=1)

                val_accuracy += losses.accuracy(scores, y)

        # Report out
        t_acc = float(train_accuracy[0] / train_accuracy[1])
        v_acc = float(val_accuracy[0] / val_accuracy[1])
        t_kn_conf = float(train_confidence[0] / train_confidence[1])
        t_un_conf = float(train_confidence[2] / train_confidence[3])
        v_kn_conf = float(val_confidence[0] / val_confidence[1])
        v_un_conf = float(val_confidence[2] / val_confidence[3])
        curr_score = float(val_confidence[0] / val_confidence[1]) + float(val_confidence[2] / val_confidence[3])

        if math.isnan(t_un_conf):
            t_un_conf = t_kn_conf
        if math.isnan(v_un_conf):
            v_un_conf = v_kn_conf

        train.report({'train_accuracy':(t_acc),
                      'val_accuracy':(v_acc),
                      'train_conf':((t_kn_conf+t_un_conf)/2),
                      'val_conf':((v_kn_conf+v_un_conf)/2),
                      'model_score':(curr_score)})
        
    print("Finished Training")

def main():

    ray.init(configure_logging=False, _temp_dir='/home/user/hkim/UZH-MT/openset-binary/_ray')
    seed = 42
    set_seeds(seed)

    if APPROACH == 'OpenSetOvR':
        param_space={
            "sigma": tune.grid_search(SIGMA), # 6 Sigmoid offset [5,6,7,8,9,10]
            "mode_param": tune.grid_search(MODE_PARAM)
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
                grace_period=40 if SCALE=='SmallScale' else 40,
                max_t=70 if SCALE == 'SmallScale' else 120,
            ),
            metric="model_score",
            mode="max",
            num_samples=1,
        ),
        run_config=RunConfig(
            progress_reporter=CLIReporter(
                max_report_frequency=300,
                print_intermediate_tables=True,  # Optional: Set to True if you want intermediate tables,
                max_column_length=20
            ),
            storage_path = '/home/user/hkim/UZH-MT/openset-binary/_ray'
        ),
        param_space=param_space,
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="model_score", mode="max")

    print(f"Best trial config: {best_result.config}\n\n")
    print("Hyperparameter Tuning Done!")

if __name__ == "__main__":
    print(f"{SCALE} {ARCH} {APPROACH} {MODE} {MODE_PARAM}")
    main()