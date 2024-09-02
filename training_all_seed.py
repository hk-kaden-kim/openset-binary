import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim

from library import architectures, tools, losses, dataset

import pathlib

import time

########################################################################
# Reference Code
# 
# Author: Manuel Günther
# Date: 2024
# Availability: https://gitlab.uzh.ch/manuel.guenther/eos-example
########################################################################

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='...TBD'
    )

    parser.add_argument("--config", "-cf", default='config/train.yaml', help="The configuration file that defines the experiment")
    parser.add_argument("--scale", "-sc", required=True, choices=['SmallScale', 'LargeScale_1', 'LargeScale_2', 'LargeScale_3'], help="Choose the scale of training dataset.")
    parser.add_argument("--arch", "-ar", required=True)
    parser.add_argument("--approach", "-ap", required=True, choices=['SoftMax', 'Garbage', 'EOS','MultiBinary'])
    parser.add_argument("--seed", "-s", default=42, nargs="+", type=int)
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()

def get_data_and_loss(args, config, epochs, seed):
    """...TBD..."""

    if args.scale == 'SmallScale':
        data = dataset.EMNIST(config.data.smallscale.root, 
                              split_ratio = config.data.smallscale.split_ratio, seed = seed,
                              convert_to_rgb = args.scale == 'SmallScale' and 'ResNet' in args.arch)
    else:
        data = dataset.IMAGENET(config.data.largescale.root, 
                                protocol_root = config.data.largescale.protocol, 
                                protocol = int(args.scale.split('_')[1]))
    
    if args.approach == "SoftMax":
        training_data, val_data, num_classes = data.get_train_set(include_negatives=False, has_background_class=False)
        loss_func=nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    
    elif args.approach =="Garbage":
        training_data, val_data, num_classes = data.get_train_set(include_negatives=True, has_background_class=True)
        loss_func=nn.CrossEntropyLoss(reduction='mean')

    elif args.approach == "EOS":
        training_data, val_data, num_classes = data.get_train_set(include_negatives=True, has_background_class=False)
        loss_func=losses.entropic_openset_loss(num_of_classes=num_classes, unkn_weight=config.loss.eos.unkn_weight)

    elif args.approach == "MultiBinary":
        training_data, val_data, num_classes = data.get_train_set(include_negatives=True, has_background_class=False)
        gt_labels = None

        is_global = sum(
            [config.loss.mbc.option == 'wclass' and config.loss.mbc.wclass_type == 'global',
             config.loss.mbc.option == 'focal' and config.loss.mbc.focal_alpha == 'global']
        )

        if is_global:
            gt_labels = dataset.get_gt_labels(training_data, gpu=args.gpu)

        loss_func=losses.multi_binary_loss(num_of_classes=num_classes, gt_labels=gt_labels, loss_config=config.loss.mbc, epochs=epochs)


    return dict(
                loss_func=loss_func,
                training_data = training_data,
                val_data = val_data,
                num_classes = num_classes
            )
    
def train(args, config, seed):

    print(f"Seed: {seed}")

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
    
    # Setting
    torch.manual_seed(seed)

    # get training data and loss function(s)
    loss_func, training_data, validation_data, num_classes = list(zip(*get_data_and_loss(args, config, epochs, seed).items()))[-1]

    results_dir = pathlib.Path(f"{args.scale}/_s{seed}/{args.arch}/{args.approach}")
    
    # if args.scale == 'LargeScale' and config.data.largescale.level > 1:
    #     results_dir = pathlib.Path(f"{args.scale}_{config.data.largescale.level}/_s{seed}/{args.arch}/{args.approach}")

    model_file = f"{results_dir}/{args.approach}.model"
    results_dir.mkdir(parents=True, exist_ok=True)

    # instantiate network and data loader
    # Add bias term at the last layer, if it is either 'Garbage' and 'MultiBinary'
    final_layer_bias = False 
    if args.approach in ['MultiBinary']:    # 'Garbage'
    # if True:
        final_layer_bias = True

    if 'LeNet_plus_plus' in args.arch:
        arch_name = 'LeNet_plus_plus'
    elif 'ResNet_18' in args.arch:
        arch_name = 'ResNet_18'
    elif 'ResNet_50' in args.arch:
        arch_name = 'ResNet_50'
    else:
        arch_name = None
    net = architectures.__dict__[arch_name](use_BG=args.approach == "Garbage",
                                            num_classes=num_classes,
                                            final_layer_bias=final_layer_bias)

    net = tools.device(net)
    
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

    # set the solver
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

    # train network
    logs_dir = results_dir/'Logs'
    writer = SummaryWriter(logs_dir)
    prev_confidence = None
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Trainig Start!")
    for epoch in range(1, epochs + 1, 1):
        t0 = time.time() # Check a duration of one epoch.

        loss_history = []
        train_accuracy = torch.zeros(2, dtype=int)
        train_magnitude = torch.zeros(2, dtype=float)
        train_confidence = torch.zeros(2, dtype=float)
        net.train()
        
        # update the network weights
        i = 0
        for x, y in train_data_loader:
            x = tools.device(x)
            y = tools.device(y)
            optimizer.zero_grad()
            logits, features = net(x)
            
            # first loss is always computed, second loss only for some loss functions
            if args.approach in ("MultiBinary"):
                loss = loss_func(logits, y, epoch)
            else:
                loss = loss_func(logits, y)

            # metrics on training set
            train_accuracy += losses.accuracy(logits, y)
            if args.approach in ("MultiBinary"):
                train_confidence += losses.multi_binary_confidence(logits, y, num_classes)
            else:
                train_confidence += losses.confidence(logits, y)
            if args.approach in ("EOS", "Objectosphere"):
                train_magnitude += losses.sphere(features, y, args.Minimum_Knowns_Magnitude if args.approach in args.approach == "Objectosphere" else None)
            # print(logits)

            loss_history.append(loss)
            loss.backward()
            optimizer.step()

            # i+=1
            # print(loss)
            # assert i < 10, f"FLAG!"
        
        # metrics on validation set
        with torch.no_grad():
            val_loss = torch.zeros(2, dtype=float)
            val_accuracy = torch.zeros(2, dtype=int)
            val_magnitude = torch.zeros(2, dtype=float)
            val_confidence = torch.zeros(2, dtype=float)
            net.eval()

            for x,y in val_data_loader:
                # predict
                x = tools.device(x)
                y = tools.device(y)
                logits, features = net(x)
                
                if args.approach in ("MultiBinary"):
                    loss = loss_func(logits, y, epoch)
                else:
                    loss = loss_func(logits, y)

                # metrics on validation set
                val_loss += torch.tensor((loss * len(y), len(y)))
                val_accuracy += losses.accuracy(logits, y)
                if args.approach in ("MultiBinary"):
                    val_confidence += losses.multi_binary_confidence(logits, y, num_classes) # Check to get the right confidence 
                else:
                    val_confidence += losses.confidence(logits, y)
                if args.approach not in ("SoftMax", "Garbage"):
                    val_magnitude += losses.sphere(features, y, args.Minimum_Knowns_Magnitude if args.approach == "Objectosphere" else None)

        # log statistics
        epoch_running_loss = torch.mean(torch.tensor(loss_history))
        writer.add_scalar('Loss/train', epoch_running_loss, epoch)
        writer.add_scalar('Loss/val', val_loss[0] / val_loss[1], epoch)
        writer.add_scalar('Acc/train', float(train_accuracy[0]) / float(train_accuracy[1]), epoch)
        writer.add_scalar('Acc/val', float(val_accuracy[0]) / float(val_accuracy[1]), epoch)
        writer.add_scalar('Conf/train', float(train_confidence[0]) / float(train_confidence[1]), epoch)
        writer.add_scalar('Conf/val', float(val_confidence[0]) / float(val_confidence[1]), epoch)
        writer.add_scalar('Mag/train', train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else 0, epoch)
        writer.add_scalar('Mag/val', val_magnitude[0] / val_magnitude[1], epoch)
        
        # save network based on confidence metric of validation set
        save_status = "NO"
        if prev_confidence is None or (val_confidence[0] > prev_confidence):
            torch.save(net.state_dict(), model_file)
            prev_confidence = val_confidence[0]
            save_status = "YES"
        
        # print some statistics
        print(f"Epoch {epoch} ({time.time()-t0:.2f}sec): "
              f"train loss {epoch_running_loss:.10f} "
              f"accuracy {float(train_accuracy[0]) / float(train_accuracy[1]):.5f} "
              f"confidence {train_confidence[0] / train_confidence[1]:.5f} "
              f"magnitude {train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else -1:.5f} -- "
              f"val loss {float(val_loss[0]) / float(val_loss[1]):.10f} "
              f"accuracy {float(val_accuracy[0]) / float(val_accuracy[1]):.5f} "
              f"confidence {val_confidence[0] / val_confidence[1]:.5f} "
              f"magnitude {val_magnitude[0] / val_magnitude[1] if val_magnitude[1] else -1:.5f} -- "
              f"Saving Model {save_status}")

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
          )

    for s in args.seed:
        train(args, config, s)
        print("Training Done!\n\n\n")