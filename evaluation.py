import torch
from torch.nn import functional as F
import numpy
import os
import time 

from library import architectures, tools, evals, dataset

import matplotlib
matplotlib.rcParams["font.size"] = 18
from matplotlib import pyplot, patches
from matplotlib.backends.backend_pdf import PdfPages

import pathlib
########################################################################
# Reference Code
# 
# Author: Manuel Günther
# Availability: https://gitlab.uzh.ch/manuel.guenther/eos-example
########################################################################

labels={
  "SoftMax" : "Plain SoftMax",
  "Garbage" : "Garbage Class",
  "EOS" : "Entropic Open-Set",
  "MultiBinary" : "Multiple Binary Classifiers",
}

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='...TBD'
    )

    parser.add_argument("--config", "-cf", default='config/eval.yaml', help="The configuration file that defines the experiment")
    parser.add_argument("--scale", "-sc", required=True, choices=['SmallScale', 'LargeScale'], help="Choose the scale of evaluation dataset.")
    parser.add_argument("--arch", "-ar", required=True)
    parser.add_argument("--approach", "-ap", nargs="+", default=list(labels.keys()), choices=list(labels.keys()), help = "Select the approaches to evaluate; non-existing models will automatically be skipped")
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()

def deep_features_plot(which, net, unkn_gt_label, pred_results, get_probs_fn, results_dir:pathlib.Path):

    print('Plot Deep Feature Space!')
    train_gt, _, train_feats, _ = pred_results['train']
    test_neg_gt, _, test_neg_feats, _ = pred_results['test_neg']
    test_unkn_gt, _, test_unkn_feats, _ = pred_results['test_unkn']

    # Deep Feature Plotting : Training Samples
    print("Training Set...")
    known_tag = train_gt != unkn_gt_label
    unknown_tag = ~known_tag

    pos_features = train_feats[known_tag]
    labels = train_gt[known_tag]
    neg_features = train_feats[unknown_tag]

    tools.viz.plotter_2D(pos_features, labels, 
                         neg_features=None, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/1_{}_train.{}')

    tools.viz.plotter_2D(pos_features, labels, 
                         neg_features=None, heat_map=True, 
                         final=True, file_name=str(results_dir)+'/2_{}_heat_train.{}',
                         prob_function=get_probs_fn, which=which, net=net, gpu=tools.get_device())

    tools.viz.plotter_2D(pos_features, labels, 
                         neg_features=neg_features, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/3_{}_train_neg.{}')
    print("Done!")

    # Deep Feature Plotting : Testing Samples (+ Negatives)
    print("Test Set with 'Known Unknown Samples'...")
    known_tag = test_neg_gt != unkn_gt_label
    unknown_tag = ~known_tag

    pos_features = test_neg_feats[known_tag]
    labels = test_neg_gt[known_tag]
    neg_features = test_neg_feats[unknown_tag]

    tools.viz.plotter_2D(pos_features, labels, 
                         neg_features=None, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/1_{}_test.{}')

    tools.viz.plotter_2D(pos_features, labels,
                         neg_features=None, heat_map=True, 
                         final=True, file_name=str(results_dir)+'/2_{}_heat_test.{}',
                         prob_function=get_probs_fn, which=which, net=net, gpu=tools.get_device())

    tools.viz.plotter_2D(pos_features, labels, 
                         neg_features=neg_features, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/3_{}_test_neg.{}'),
    print("Done!")

    # Deep Feature Plotting : Testing Samples (+ Unknowns)
    print("Test Set with 'Unknown Unknown Samples'...")
    known_tag = test_unkn_gt != unkn_gt_label
    unknown_tag = ~known_tag

    pos_features = test_unkn_feats[known_tag]
    labels = test_unkn_gt[known_tag]
    neg_features = test_unkn_feats[unknown_tag]

    tools.viz.plotter_2D(pos_features, labels, 
                         neg_features=neg_features, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/3_{}_test_unkn.{}')

    print("Done!\n")

def evaluate(args, config):

    # load dataset
    if args.scale == 'SmallScale':
        data = dataset.EMNIST(config.data.smallscale.root,
                              split_ratio = 0.8, seed = config.seed,
                              convert_to_rgb=args.scale == 'SmallScale' and 'ResNet' in args.arch)
    else:
        data = dataset.IMAGENET(config.data.largescale.root,
                                protocol_root = config.data.largescale.protocol, 
                                protocol = config.data.largescale.level)

    # Save or Plot results
    results = {}
    root = pathlib.Path(f"{args.scale}/eval_{args.arch}")

    # if args.scale == 'SmallScale' and config.arch.force_fc_dim == 2:
    #     root = pathlib.Path(f"{args.scale}_fc_dim_2/eval_{args.arch}")
    
    if args.scale == 'LargeScale' and config.data.largescale.level > 1:
        root = pathlib.Path(f"{args.scale}_{config.data.largescale.level}/eval_{args.arch}")
    root.mkdir(parents=True, exist_ok=True)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Configuration Details \n"
        f"Model Root: {config.arch.model_root}\n"
        f"Save Predictions: {config.pred_save==1}\n"
        f"Save Predictions: {config.oscr_save==1}\n"
          )

    for which in args.approach:

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print (f"Evaluation: {which}\n"
               f"Execution Time: {time.strftime('%d %b %Y %H:%M:%S')}\n")

        # Set the variables
        results_dir = root.joinpath(which)
        results_dir.mkdir(parents=True, exist_ok=True)

        pred_results = {'train':None, 'test_neg':None, 'test_unkn':None, 'test_all':None}

        if args.scale == 'SmallScale':
            batch_size = config.batch_size.smallscale
        else:
            batch_size = config.batch_size.largescale
        print(f"Batch Size: {batch_size}\n"
              f"Results: {results_dir}\n")

        # Load evaluation dataset
        if which == 'Garbage':
            train_set_neg, _, num_classes = data.get_train_set(include_negatives=True, has_background_class=True)
            _, test_set_neg, test_set_unkn = data.get_test_set(has_background_class=True)
            unkn_gt_label = num_classes
        else:
            train_set_neg, _, num_classes = data.get_train_set(include_negatives=True, has_background_class=False)
            _, test_set_neg, test_set_unkn = data.get_test_set(has_background_class=False)
            unkn_gt_label = -1
        
        # Load weights of the model
        net = evals.load_network(args, config, which, num_classes)
        if net is None:
            print(f"Weights are not loaded on the network!\n{which} Evaluation Terminated\n")
            continue
        
        print("Execute predictions!")
        if args.scale == 'SmallScale':
            print(f"{time.strftime('%H:%M:%S')} Training Set...")
            pred_results['train'] = evals.extract(train_set_neg, net, batch_size, is_verbose=True)
            print(f"{time.strftime('%H:%M:%S')} Done!")
        print(f"{time.strftime('%H:%M:%S')} Test Set with 'Known Unknown Samples'...")
        pred_results['test_neg'] = evals.extract(test_set_neg, net,  batch_size, is_verbose=True)
        print(f"{time.strftime('%H:%M:%S')} Done!")
        print(f"{time.strftime('%H:%M:%S')} Test Set with 'Unknown Unknown Samples'...")
        pred_results['test_unkn'] = evals.extract(test_set_unkn, net, batch_size, is_verbose=True)
        print(f"{time.strftime('%H:%M:%S')} Done!")
        print()

        # Calculate Probs
        if which == "MultiBinary":
            if args.scale == 'SmallScale':
                train_probs = F.sigmoid(torch.tensor(pred_results['train'][1])).detach().numpy()
            test_neg_probs = F.sigmoid(torch.tensor(pred_results['test_neg'][1])).detach().numpy()
            test_unkn_probs  = F.sigmoid(torch.tensor(pred_results['test_unkn'][1])).detach().numpy()
        else:
            if args.scale == 'SmallScale':
                train_probs = F.softmax(torch.tensor(pred_results['train'][1]), dim=1).detach().numpy()
            test_neg_probs = F.softmax(torch.tensor(pred_results['test_neg'][1]), dim=1).detach().numpy()
            test_unkn_probs  = F.softmax(torch.tensor(pred_results['test_unkn'][1]), dim=1).detach().numpy()

        # # remove the labels for the unknown class in case of Garbage Class
        # if which == "Garbage":
        #     test_neg_probs = test_neg_probs[:,:-1]
        #     test_unkn_probs = test_unkn_probs[:,:-1]
        
        if args.scale == 'SmallScale':
            pred_results['train'].append(train_probs)
        pred_results['test_neg'].append(test_neg_probs)
        pred_results['test_unkn'].append(test_unkn_probs)

        # # remove the labels for the unknown class in case of Garbage Class
        if config.pred_save:
            evals.eval_pred_save(pred_results, results_dir.joinpath('pred'), save_feats = 'LeNet_plus_plus' in args.arch)

        if which == "Garbage":
            pred_results['test_neg'][-1] = pred_results['test_neg'][-1][:,:-1]
            pred_results['test_unkn'][-1] = pred_results['test_unkn'][-1][:,:-1]

        if args.scale == 'SmallScale':
            deep_features_plot(which, net, 
                               results_dir=results_dir, 
                               unkn_gt_label=unkn_gt_label, 
                               pred_results=pred_results, 
                               get_probs_fn=tools.viz.get_probs)


        print('Get OSCR results')
        print("Test Set with 'Known Unknown Samples'...")
        ccr, fpr_neg = evals.get_oscr_curve(pred_results['test_neg'][0], pred_results['test_neg'][3]
                                            , unkn_gt_label, at_fpr=None)
        print("Test Set with 'Unknown Unknown Samples'...")
        _, fpr_unkn = evals.get_oscr_curve(pred_results['test_unkn'][0], pred_results['test_unkn'][3]
                                           , unkn_gt_label,  at_fpr=None)
        print('Done!\n')

        results[which] = (ccr, fpr_neg, fpr_unkn)

        if config.oscr_save:
            evals.oscr_save(ccr, fpr_neg, fpr_unkn, results_dir.joinpath('oscr'))

        torch.cuda.empty_cache()
        print('Release Unoccupied cache in GPU!')

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Final OSCR Plot for {args.approach}")
    try:
        # plot with known unknowns
        pyplot.figure(figsize=(10,5))
        for which, res in results.items():
            pyplot.semilogx(res[1], res[0], label=labels[which])
        pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("Correct Classification Rate")
        pyplot.title("Negative Set")
        pyplot.tight_layout()
        pyplot.savefig(root.joinpath('oscr_neg.png'), bbox_inches="tight") 
        
        # plot with unknown unknowns
        pyplot.figure(figsize=(10,5))
        for which, res in results.items():
            pyplot.semilogx(res[2], res[0], label=labels[which])
        pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("Correct Classification Rate")
        pyplot.title("Unknown Set")
        pyplot.tight_layout()
        pyplot.savefig(root.joinpath('oscr_unkn.png'), bbox_inches="tight") 

    finally:
        print('Done!\n')

if __name__ == '__main__':

    args = command_line_options()
    config = tools.load_yaml(args.config)

    if args.gpu is not None and torch.cuda.is_available():
        tools.set_device_gpu(args.gpu)
    else:
        print("Running in CPU mode, might be slow")
        tools.set_device_cpu()

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Execution Time: {time.strftime('%d %b %Y %H:%M:%S')} \n"
        f"GPU: {args.gpu} \n"
        f"Dataset Scale: {args.scale} \n"
        f"Architecture: {args.arch} \n"
        f"Approach: {args.approach} \n"
        f"Configuration: {args.config} \n"
          )
    
    evaluate(args, config)
    print("Evaluation Done!")

