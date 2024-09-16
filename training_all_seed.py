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

########################################################################
# Version 2
# - (DONE) Validation Set confidence matrics IN
# - (DONE) Garbage : Class weight ADDED
########################################################################


########################################################################
# Reference Code
# 
# Author: Manuel Günther
# Date: 2024
# Availability: https://gitlab.uzh.ch/manuel.guenther/eos-example
########################################################################


def set_seeds(seed):
    """ Sets the seed for different sources of randomness.

    Args:
        seed(int): Integer
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='...TBD'
    )

    parser.add_argument("--config", "-cf", default='config/train.yaml', help="The configuration file that defines the experiment")
    parser.add_argument("--scale", "-sc", required=True, choices=['SmallScale', 'LargeScale_1', 'LargeScale_2', 'LargeScale_3'], help="Choose the scale of training dataset.")
    parser.add_argument("--arch", "-ar", required=True)
    parser.add_argument("--approach", "-ap", required=True, choices=['SoftMax', 'Garbage', 'EOS', 'OvR', 'OpenSetOvR'])
    parser.add_argument("--seed", "-s", default=42, nargs="+", type=int)
    parser.add_argument("--debug", "-dg", default=0, type=int)
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()

def get_data_and_loss(args, config, epochs, seed):
    """...TBD..."""

    if args.scale == 'SmallScale':
        data = dataset.EMNIST(config.data.smallscale.root, 
                              split_ratio = config.data.smallscale.split_ratio, seed = seed)
    else:
        data = dataset.IMAGENET(config.data.largescale.root, 
                                protocol_root = config.data.largescale.protocol, 
                                protocol = int(args.scale.split('_')[1]),
                                is_verbose=True)
    
    if args.approach == "SoftMax":
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, has_background_class=False, size_train_negatives=0)
        loss_func=nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    
    elif args.approach =="Garbage":
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, has_background_class=True, size_train_negatives=config.data.train_neg_size)
        c_weights = dataset.calc_class_weight(training_data, gpu=args.gpu)
        print(f"Class weight : {c_weights}")
        if len(c_weights) != num_classes+1:
            print(f"Add the last weight. {tools.device(torch.Tensor([1]))}")
            c_weights = torch.cat((c_weights, tools.device(torch.Tensor([1]))))
            print(c_weights)
        loss_func=nn.CrossEntropyLoss(weight = c_weights, reduction='mean')

    elif args.approach == "EOS":
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, has_background_class=False, size_train_negatives=config.data.train_neg_size)
        loss_func=losses.entropic_openset_loss(num_of_classes=num_classes, unkn_weight=config.loss.eos.unkn_weight)

    elif args.approach == 'OvR':
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, has_background_class=False, size_train_negatives=config.data.train_neg_size)
        loss_func=losses.OvRLoss(num_of_classes=num_classes, loss_config=config.loss.ovr, training_data=training_data)

    elif args.approach == "OpenSetOvR":
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, has_background_class=False, size_train_negatives=config.data.train_neg_size)
        loss_func=losses.OSOvRLoss(num_of_classes=num_classes, loss_config=config.loss.osovr, training_data=training_data)

    return dict(
                loss_func=loss_func,
                training_data = training_data,
                val_data = val_data,
                num_classes = num_classes
            )

def train(args, net, optimizer, train_data_loader, loss_func, epoch, num_classes):

    loss_history = []
    train_accuracy = torch.zeros(2, dtype=int)
    train_confidence = torch.zeros(4, dtype=float)
    net.train()
    
    # update the network weights
    num_batch = 0
    for x, y in train_data_loader:
        x = tools.device(x)
        y = tools.device(y)
        optimizer.zero_grad()
        logits, _ = net(x)
            
        # first loss is always computed, second loss only for some loss functions
        if args.approach == "OpenSetOvR":
            loss = loss_func(logits, y, last_layer_weights = net.fc2.weight.data)
        else:
            loss = loss_func(logits, y)

        if args.approach == "OvR":
            scores = F.sigmoid(logits)
            train_confidence += losses.confidence(scores, y,
                                                    offset = 0.,
                                                    unknown_class = -1,
                                                    last_valid_class = None,)
        elif args.approach == 'OpenSetOvR':
            scores = loss_func.osovr_act(logits, net.fc2.weight.data)
            train_confidence += losses.confidence(scores, y,
                                                    offset = 0.,
                                                    unknown_class = -1,
                                                    last_valid_class = None,)
        else:
            scores = torch.nn.functional.softmax(logits, dim=1)
            if args.approach == "Garbage":
                train_confidence += losses.confidence(scores, y,
                                                        offset = 0.,
                                                        unknown_class = num_classes,
                                                        last_valid_class = -1,)
            else:
                train_confidence += losses.confidence(scores, y,
                                                        offset = 1. / num_classes,
                                                        unknown_class = -1,
                                                        last_valid_class = None,)
        train_accuracy += losses.accuracy(scores, y)

        loss_history.append(loss)
        loss.backward()
        optimizer.step()

        num_batch += 1

    return loss_history, train_accuracy, train_confidence

def validate(args, net, val_data_loader, loss_func, epoch, num_classes):

    with torch.no_grad():
        val_loss = torch.zeros(2, dtype=float)
        val_accuracy = torch.zeros(2, dtype=int)
        val_confidence = torch.zeros(4, dtype=float)
        net.eval()

        num_batch = 0
        for x,y in val_data_loader:
            # predict
            x = tools.device(x)
            y = tools.device(y)
            logits, _ = net(x)

            if args.approach == "OpenSetOvR":
                loss = loss_func(logits, y, last_layer_weights = net.fc2.weight.data)
            else:
                loss = loss_func(logits, y)

            # metrics on validation set
            if ~torch.isnan(loss):
                val_loss += torch.tensor((loss * len(y), len(y)))
                
            if args.approach == "OvR":
                scores = F.sigmoid(logits)
                val_confidence += losses.confidence(scores, y,
                                                        offset = 0.,
                                                        unknown_class = -1,
                                                        last_valid_class = None,)
            elif args.approach == 'OpenSetOvR':
                scores = loss_func.osovr_act(logits, net.fc2.weight.data)
                val_confidence += losses.confidence(scores, y,
                                                        offset = 0.,
                                                        unknown_class = -1,
                                                        last_valid_class = None,)
            else:
                scores = torch.nn.functional.softmax(logits, dim=1)
                if args.approach == "Garbage":
                    val_confidence += losses.confidence(scores, y,
                                                            offset = 0.,
                                                            unknown_class = num_classes,
                                                            last_valid_class = -1,)
                else:
                    val_confidence += losses.confidence(scores, y,
                                                            offset = 1. / num_classes,
                                                            unknown_class = -1,
                                                            last_valid_class = None,)
            val_accuracy += losses.accuracy(scores, y)

            num_batch += 1
    
    return val_loss, val_accuracy, val_confidence

def worker(args, config, seed):

    print(f"Seed: {seed}")
    set_seeds(seed)
    BEST_SCORE = 0

    # Load Training Parameters
    if args.scale == 'SmallScale':
        batch_size = config.batch_size.smallscale
        epochs = config.epochs.smallscale
    else:
        batch_size = config.batch_size.largescale
        epochs = config.epochs.largescale
    num_workers = config.num_workers
    solver = config.opt.solver
    lr = config.opt.lr
    lr_decay = config.opt.decay
    lr_gamma = config.opt.gamma

    # SETTING. Directory
    results_dir = pathlib.Path(f"{args.scale}/_s{seed}/{args.arch}/{args.approach}")
    model_file = f"{results_dir}/{args.approach}.model"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir/'Logs'
    writer = SummaryWriter(logs_dir)

    # SETTING. Dataset and Loss function
    loss_func, training_data, validation_data, num_classes = list(zip(*get_data_and_loss(args, config, epochs, seed).items()))[-1]
    
    # SETTING. DataLoader
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

    # SETTING. NN Architecture
    if 'LeNet' in args.arch:
        arch_name = 'LeNet'
        if 'plus_plus' in args.arch:
            arch_name = 'LeNet_plus_plus'
    elif 'ResNet_18' in args.arch:
        arch_name = 'ResNet_18'
    elif 'ResNet_50' in args.arch:
        arch_name = 'ResNet_50'
    else:
        arch_name = None
    net = architectures.__dict__[arch_name](use_BG=args.approach == "Garbage",
                                            # init_weights=args.approach == "OpenSetOvR",
                                            init_weights=False,
                                            num_classes=num_classes,
                                            final_layer_bias=False)
    net = tools.device(net)
    if args.debug:
        model_file_history = model_file.replace('.model', f"_0.model")
        torch.save(net.state_dict(), model_file_history)

    # SETTING. Optimizer
    if solver == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    if lr_decay > 0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=lr_gamma)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Configuration Details \n"
        f"Train: {len(training_data)}\tVal: {len(validation_data)}\n"
        f"Batch Size: {batch_size} \n"
        f"Epochs: {epochs} \n"
        f"Solver: {solver} \n"
        f"Learning Rate: {lr} (Scheduler: {lr_decay > 0}) \n"
        f"Results: {model_file} \n"
          )

    # TRAINING
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Trainig Start!")

    for epoch in range(1, epochs + 1, 1):
        t0 = time.time() # Check a duration of one epoch.

        loss_history, train_accuracy, train_confidence = train(args, net, 
                                                                  optimizer, 
                                                                  train_data_loader, 
                                                                  loss_func, 
                                                                  epoch, num_classes)
        
        # metrics on validation set
        val_loss, val_accuracy, val_confidence = validate(args, net, 
                                                             val_data_loader, 
                                                             loss_func, 
                                                             epoch, num_classes)

        # save network based on confidence metric of validation set
        save_status = "NO"
        if config.data.train_neg_size == 0:
            curr_score = float(val_confidence[0] / val_confidence[1])
        else:
            curr_score = float(val_confidence[0] / val_confidence[1]) + float(val_confidence[2] / val_confidence[3])
        if curr_score > BEST_SCORE:
            torch.save(net.state_dict(), model_file)
            BEST_SCORE = curr_score
            save_status = "YES"

            # Debugging purpose
            if args.debug:
                model_file_history = model_file.replace('.model', f"_{epoch}.model")
                torch.save(net.state_dict(), model_file_history)
            

        # log statistics
        epoch_running_loss = torch.mean(torch.tensor(loss_history))
        writer.add_scalar('Loss/train', epoch_running_loss, epoch)
        writer.add_scalar('Loss/val', val_loss[0] / val_loss[1], epoch)
        writer.add_scalar('Acc/train', float(train_accuracy[0] / train_accuracy[1]), epoch)
        writer.add_scalar('Acc/val', float(val_accuracy[0] / val_accuracy[1]), epoch)
        writer.add_scalar('Conf/train_kn', float(train_confidence[0] / train_confidence[1]), epoch)
        writer.add_scalar('Conf/train_neg', float(train_confidence[2] / train_confidence[3]), epoch)
        writer.add_scalar('Conf/val_kn', float(val_confidence[0] / val_confidence[1]), epoch)
        writer.add_scalar('Conf/val_neg', float(val_confidence[2] / val_confidence[3]), epoch)
        
        # print some statistics
        print(f"Epoch {epoch} ({time.time()-t0:.2f}sec): "
              f"TRAIN SET -- "
              f"Loss {epoch_running_loss:.5f} "
              f"Acc {float(train_accuracy[0] / train_accuracy[1]):.5f} "
              f"KnConf {float(train_confidence[0] / train_confidence[1]):.5f} "
              f"UnConf {float(train_confidence[2] / train_confidence[3]):.5f} "
              f"VALIDATION SET -- "
              f"Loss {float(val_loss[0] / val_loss[1]):.5f} "
              f"Acc {float(val_accuracy[0] / val_accuracy[1]):.5f} "
              f"KnConf {float(val_confidence[0] / val_confidence[1]):.5f} "
              f"UnConf {float(val_confidence[2] / val_confidence[3]):.5f} "
              f"SAVING MODEL -- {curr_score:.3f} {save_status}")

        if lr_decay > 0:
            scheduler.step()
        

if __name__ == "__main__":

    args = command_line_options()
    config = tools.load_yaml(args.config)

    if args.gpu is not None and torch.cuda.is_available():
        tools.set_device_gpu(args.gpu)
    else:
        print("Running in CPU mode, training might be slow")
        tools.set_device_cpu()

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Execution Time: {time.strftime('%d %b %Y %H:%M:%S')} \n"
        f"GPU: {args.gpu} \n"
        f"Dataset Scale: {args.scale} \n"
        f"Architecture: {args.arch} \n"
        f"Approach: {args.approach} \n"
        f"Configuration: {args.config} \n"
        f"Seed: {args.seed}\n"
        f"Debug mode: {True if args.debug==1 else False}"
          )

    for s in args.seed:
        worker(args, config, s)
        print("Training Done!\n\n\n")
    
    print("All training done!")