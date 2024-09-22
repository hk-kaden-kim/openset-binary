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
import matplotlib.pyplot as plt

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

def check_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().tolist())
    # plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    return ave_grads, layers

def check_fc2_weights(net,):
    return net.fc2.weight.tolist()

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

def get_data_and_loss(args, config, arch_name, seed):
    """...TBD..."""

    if args.scale == 'SmallScale':
        data = dataset.EMNIST(config.data.smallscale.root, 
                              split_ratio = config.data.smallscale.split_ratio, 
                              label_filter=config.data.smallscale.label_filter,
                              seed = seed, convert_to_rgb='ResNet' in args.arch)
    else:
        data = dataset.IMAGENET(config.data.largescale.root, 
                                protocol_root = config.data.largescale.protocol, 
                                protocol = int(args.scale.split('_')[1]),
                                is_verbose=True)
    
    if args.approach == "SoftMax":
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, has_background_class=False, size_train_negatives=config.data.train_neg_size)
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
        loss_func=losses.OvRLoss(num_of_classes=num_classes, mode=config.loss.ovr.mode, training_data=training_data)

    elif args.approach == "OpenSetOvR":
        training_data, val_data, num_classes = data.get_train_set(is_verbose=True, has_background_class=False, size_train_negatives=config.data.train_neg_size)
        loss_func=losses.OSOvRLoss(num_of_classes=num_classes, sigma=config.loss.osovr.sigma.dict()[arch_name], mode=config.loss.osovr.mode, training_data=training_data)

    return dict(
                loss_func=loss_func,
                training_data = training_data,
                val_data = val_data,
                num_classes = num_classes
            )


# Taken from:
# https://github.com/Lance0218/Pytorch-DistributedDataParallel-Training-Tricks/
class EarlyStopping:
    """ Stops the training if validation loss/metrics doesn't improve after a given patience"""
    def __init__(self, patience=100, delta=0):
        """
        Args:
            patience(int): How long wait after last time validation loss improved. Default: 100
            delta(float): Minimum change in the monitored quantity to qualify as an improvement
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        if loss is True:
            score = -metrics
        else:
            score = metrics

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train(args, net, optimizer, train_data_loader, loss_func, num_classes, debug=False):

    loss_history = []
    train_accuracy = torch.zeros(2, dtype=int)
    train_confidence = torch.zeros(4, dtype=float)
    net.train()
    
    # update the network weights
    num_batch = 0
    grad_results, fc2_weights_results, layers = [], [], []
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

        # DEBUG : Check for vanishing or exploding gradient
        if debug:
            batch_grad, layers = check_grad_flow(net.named_parameters())
            batch_fc2_weights = check_fc2_weights(net)   

            grad_results.append(batch_grad)
            fc2_weights_results.append(batch_fc2_weights)
        
        optimizer.step()
        num_batch += 1

    debug_results = (grad_results, layers, fc2_weights_results)

    return loss_history, train_accuracy, train_confidence, debug_results

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
    if 'LeNet' in args.arch:
        arch_name = 'LeNet'
        if 'plus_plus' in args.arch:
            arch_name = 'LeNet_plus_plus'
    elif 'ResNet_50' in args.arch:
        arch_name = 'ResNet_50'
    else:
        arch_name = None

    # SETTING. Directory
    results_dir = pathlib.Path(f"{args.scale}/_s{seed}/{args.arch}/{args.approach}")
    model_file = f"{results_dir}/{args.approach}.model"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir/'Logs'
    writer = SummaryWriter(logs_dir)

    # SETTING. Dataset and Loss function
    loss_func, training_data, validation_data, num_classes = list(zip(*get_data_and_loss(args, config, arch_name, seed).items()))[-1]
    
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
    net = architectures.__dict__[arch_name](use_BG=args.approach == "Garbage",
                                            num_classes=num_classes,
                                            final_layer_bias=False,
                                            feat_dim=config.arch.feat_dim,
                                            is_osovr=args.approach == "OpenSetOvR")
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


    # # SETTING. Early Stopping
    # if args.scale == 'SmallScale':
    #     early_stopping = None
    # else:
    #     early_stopping = EarlyStopping(patience=5)

    # TRAINING
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Trainig Start!")

    for epoch in range(1, epochs + 1, 1):
        t0 = time.time() # Check a duration of one epoch.
        loss_history, train_accuracy, train_confidence, debug_info = train(args, net, 
                                                                  optimizer, 
                                                                  train_data_loader, 
                                                                  loss_func, num_classes, debug=config.training_debug)
        # print(net.fc2.weight, net.fc2.weight.shape)
        # w = net.fc2.weight ** 2
        # print(torch.sum(w, dim=0), torch.sum(w, dim=1))
        # assert False, "Terminated"

        # DEBUG : for gradient check
        if config.training_debug:
            debug_results_dir = results_dir/'Debug'
            debug_results_dir.mkdir(parents=True, exist_ok=True)
            np.save(debug_results_dir/f'grad_results_{epoch}.npy',np.array(debug_info[0]))
            np.save(debug_results_dir/f'layers.npy',np.array(debug_info[1]))
            np.save(debug_results_dir/f'fc2_weights_results_{epoch}.npy',np.array(debug_info[2]))
            print("Training debug info save done!")

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

        # # Early stopping
        # early_stopping(metrics=curr_score, loss=False)
        # if early_stopping.early_stop:
        #     logger.info("early stop")
        #     break

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

    # ARCH = args.arch
    # for s in args.seed:
    #     for item in [6,8,10]:
    #         config.loss.osovr.sigma.dict()['ResNet_50'] = item
    #         if item != -1:
    #             args.arch = ARCH + f'_{item}'
    #         else:
    #             args.arch = ARCH
    #         worker(args, config, s)
    #         print("Training Done!\n\n\n")

    print("All training done!")