# TODO: Check deep feature space visualization!
# If it takes lots of time, just plot OSCR CURVE for ResNet18/50 EMNIST and ResNet18 ImageNet

import torch
from torch.nn import functional as F
import numpy
import os

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
  "Objectosphere" : "Objectosphere",
  "MultiBinary" : "Multiple Binary Classifiers",
}

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the evaluation script for all MNIST experiments. \
                    Where applicable roman letters are used as Known Unknowns. \
                    During training model with best performance on validation set in the no_of_epochs is used.'
    )

    parser.add_argument("--approaches", "-a", nargs="+", default=list(labels.keys()), choices=list(labels.keys()), help = "Select the approaches to evaluate; non-existing models will automatically be skipped")
    parser.add_argument("--arch", '-ar', default='LeNet_plus_plus', choices=['LeNet_plus_plus'])
    parser.add_argument("--dataset", "-dt", default ="SmallScale", help="Choose the scale of training dataset.")
    parser.add_argument("--dataset_root", "-rt", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--model_root", "-mr", default ="/tmp", help="Select the directory where models are stored.")
    parser.add_argument("--protocol_root", "-pr", default ="/tmp", help="Select the directory where LargeScale Protocol are stored.")
    parser.add_argument("--protocol", "-p", default =1, help="Select the LargeScale Protocol.")
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()

def deep_features_plot(which, net, results_dir:pathlib.Path, unkn_gt_label, pred_results, get_probs_fn):

    train_gt, _, train_feats, _ = pred_results['train']
    test_neg_gt, _, test_neg_feats, _ = pred_results['test_neg']
    test_unkn_gt, _, test_unkn_feats, _ = pred_results['test_unkn']

    data = train_feats[train_gt != unkn_gt_label]
    data_labels = train_gt[train_gt != unkn_gt_label]
    neg_features = train_feats[train_gt == unkn_gt_label]

    tools.viz.plotter_2D(data, data_labels, neg_features=None,
                         final=True, heat_map=False, file_name=str(results_dir)+'/1_train_{}.{}')

    tools.viz.plotter_2D(data, data_labels, neg_features=None,
                         final=True, heat_map=True, file_name=str(results_dir)+'/2_train_{}_heat.{}',
                         prob_function=get_probs_fn, which=which, net=net, gpu=args.gpu)

    tools.viz.plotter_2D(data, data_labels, neg_features=neg_features,
                         final=True, heat_map=False, file_name=str(results_dir)+'/3_train_{}_neg.{}')

    data = test_neg_feats[test_neg_gt != unkn_gt_label]
    data_labels = test_neg_gt[test_neg_gt != unkn_gt_label]
    neg_features = test_neg_feats[test_neg_gt == unkn_gt_label]

    tools.viz.plotter_2D(data, data_labels, neg_features=None,
                         final=True, heat_map=False, file_name=str(results_dir)+'/1_test_{}.{}')

    tools.viz.plotter_2D(data, data_labels, neg_features=None,
                         final=True, heat_map=True, file_name=str(results_dir)+'/2_test_{}_heat.{}',
                         prob_function=get_probs_fn, which=which, net=net, gpu=args.gpu)

    tools.viz.plotter_2D(data, data_labels, neg_features=neg_features,
                         final=True, heat_map=False, file_name=str(results_dir)+'/3_test_{}_neg.{}'),

    data = test_unkn_feats[test_unkn_gt != unkn_gt_label]
    data_labels = test_unkn_gt[test_unkn_gt != unkn_gt_label]
    neg_features = test_unkn_feats[test_unkn_gt == unkn_gt_label]

    tools.viz.plotter_2D(data, data_labels, neg_features=neg_features,
                         final=True, heat_map=False, file_name=str(results_dir)+'/3_test_{}_unkn.{}', gpu=args.gpu)


def evaluate(args):

    # networks
    networks = {
        which: evals.load_network(args, which, args.model_root) for which in args.approaches
    }

    # load dataset
    if args.dataset == 'SmallScale':
        data = dataset.EMNIST(args.dataset_root)
    else:
        data = dataset.IMAGENET(args.dataset_root, args.protocol_root, args.protocol)

    # Save or Plot results of the prediction
    results = {}
    root = pathlib.Path(f"eval_{args.arch}")
    root.mkdir(parents=True, exist_ok=True)
    for which, net in networks.items():
        if net is None:
            print('net is none')
            continue
        print ("Evaluating", which)
        results_dir = root.joinpath(which)
        results_dir.mkdir(parents=True, exist_ok=True)

        if which == 'SoftMax':
            train_set, val_set = data.get_train_set(split_ratio=0.8, include_negatives=True, has_background_class=False)
            test_set_all, test_set_neg, test_set_unkn = data.get_test_set(has_background_class=False)
        elif which == 'Garbage':
            train_set, val_set = data.get_train_set(split_ratio=0.8, include_negatives=True, has_background_class=True)
            test_set_all, test_set_neg, test_set_unkn = data.get_test_set(has_background_class=True)
        else:
            train_set, val_set = data.get_train_set(split_ratio=0.8, include_negatives=True, has_background_class=False)
            test_set_all, test_set_neg, test_set_unkn = data.get_test_set(has_background_class=False)
        # print(train_set.datasets[0].dataset.targets)
        # print(torch.unique(train_set.datasets[0].dataset.targets))
        ################################################
        # Prediction
        ################################################
        pred_results = {'train':None, 'test_neg':None, 'test_unkn':None, 'test_all':None}
        pred_results['train'] = evals.extract(train_set, net)
        pred_results['test_neg'] = evals.extract(test_set_neg, net)
        pred_results['test_unkn'] = evals.extract(test_set_unkn, net)
        # print(numpy.unique(pred_results['test_neg'][0]))
        # assert False

        # Calculate Probs
        if args.approaches == "MultiBinary":
            train_probs = F.sigmoid(torch.tensor(pred_results['train'][1])).detach().numpy()
            test_neg_probs = F.sigmoid(torch.tensor(pred_results['test_neg'][1])).detach().numpy()
            test_unkn_probs  = F.sigmoid(torch.tensor(pred_results['test_unkn'][1])).detach().numpy()
        else:
            train_probs = F.softmax(torch.tensor(pred_results['train'][1]), dim=1).detach().numpy()
            test_neg_probs = F.softmax(torch.tensor(pred_results['test_neg'][1]), dim=1).detach().numpy()
            test_unkn_probs  = F.softmax(torch.tensor(pred_results['test_unkn'][1]), dim=1).detach().numpy()

        # remove the labels for the unknown class in case of Garbage Class
        if args.approaches == "Garbage":
            test_neg_probs = test_neg_probs[:,:-1]
            test_unkn_probs = test_unkn_probs[:,:-1]
            unkn_gt_label = 10  # Change the lable of unkn gt
        else:
            unkn_gt_label = -1

        pred_results['train'].append(train_probs)
        pred_results['test_neg'].append(test_neg_probs)
        pred_results['test_unkn'].append(test_unkn_probs)

        ################################################
        # Deep Feature Plots
        ################################################
        if args.arch == 'LeNet_plus_plus':
            deep_features_plot(which, net, results_dir, unkn_gt_label, pred_results, tools.viz.get_probs)

        ################################################
        # Get OSCR
        ################################################
        # get ccr and fpr by varying a threshold
        print("Test set : Positives + Negatives")
        ccr, fpr_neg = evals.get_oscr_curve(pred_results['test_neg'][0], pred_results['test_neg'][3]
                                            , unkn_gt_label)
        print("Test set : Positives + Unknowns")
        _, fpr_unkn = evals.get_oscr_curve(pred_results['test_unkn'][0], pred_results['test_unkn'][3]
                                           , unkn_gt_label)
        print()
        
        results[which] = (ccr, fpr_neg, fpr_unkn)

    ################################################
    # OSCR Plots
    ################################################
    try:
        # plot with known unknowns (letters 1:13)
        pyplot.figure()
        for which, res in results.items():
            pyplot.semilogx(res[1], res[0], label=labels[which])
        pyplot.legend()
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("Correct Classification Rate")
        pyplot.title("Negative Set")
        pyplot.tight_layout()
        pyplot.savefig(root.joinpath('oscr_neg.png'), bbox_inches="tight") 
        # pdf.savefig(bbox_inches='tight', pad_inches=0)
        
        # plot with unknown unknowns (letters 14:26)
        pyplot.figure()
        for which, res in results.items():
            pyplot.semilogx(res[2], res[0], label=labels[which])
        pyplot.legend()
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("Correct Classification Rate")
        pyplot.title("Unknown Set")
        pyplot.tight_layout()
        pyplot.savefig(root.joinpath('oscr_unkn.png'), bbox_inches="tight") 
        # pdf.savefig(bbox_inches='tight', pad_inches=0)

    finally:
        print('Done!')

if __name__ == '__main__':

    args = command_line_options()
    if args.gpu is not None and torch.cuda.is_available():
        tools.set_device_gpu(args.gpu)
    else:
        print("Running in CPU mode, might be slow")
        tools.set_device_cpu()

    evaluate(args)

