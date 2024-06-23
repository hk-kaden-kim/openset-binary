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
    parser.add_argument("--arch", '-ar', default='LeNet_plus_plus', choices=['LeNet_plus_plus', 'ResNet_18', 'ResNet_50'])
    parser.add_argument("--dataset", "-dt", default ="SmallScale", help="Choose the scale of training dataset.")
    parser.add_argument("--dataset_root", "-rt", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--model_root", "-mr", default ="/tmp", help="Select the directory where models are stored.")
    parser.add_argument("--protocol_root", "-pr", default ="/tmp", help="Select the directory where LargeScale Protocol are stored.")
    parser.add_argument("--protocol", "-p", default =1, help="Select the LargeScale Protocol.")
    parser.add_argument('--batch_size', "-b", default =2048, help='Batch_Size', action="store", type=int)
    parser.add_argument('--pred_save', "-sp", default =0, help='Save predictions npy if the value is 1')
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()

def deep_features_plot(which, net, unkn_gt_label, pred_results, get_probs_fn, results_dir:pathlib.Path):
    
    train_gt, _, train_feats, _ = pred_results['train']
    test_neg_gt, _, test_neg_feats, _ = pred_results['test_neg']
    test_unkn_gt, _, test_unkn_feats, _ = pred_results['test_unkn']

    #################################################################
    # Deep Feature Plotting : Training Samples
    #################################################################
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

    #################################################################
    # Deep Feature Plotting : Testing Samples (+ Negatives)
    #################################################################
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

    #################################################################
    # Deep Feature Plotting : Testing Samples (+ Unknowns)
    #################################################################
    known_tag = test_unkn_gt != unkn_gt_label
    unknown_tag = ~known_tag

    pos_features = test_unkn_feats[known_tag]
    labels = test_unkn_gt[known_tag]
    neg_features = test_unkn_feats[unknown_tag]

    tools.viz.plotter_2D(pos_features, labels, 
                         neg_features=neg_features, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/3_{}_test_unkn.{}')

def evaluate(args):

    # load dataset
    if args.dataset == 'SmallScale':
        data = dataset.EMNIST(args.dataset_root, convert_to_rgb=args.dataset == 'SmallScale' and 'ResNet' in args.arch)
    else:
        data = dataset.IMAGENET(args.dataset_root, args.protocol_root, args.protocol)

    # Save or Plot results
    results = {}
    root = pathlib.Path(f"eval_{args.arch}")
    root.mkdir(parents=True, exist_ok=True)
    # for which, net in networks.items():
    for which in args.approaches:

        print ("Evaluating", which)
        if args.arch == 'LeNet_plus_plus':
            results_dir = root.joinpath(which)
            results_dir.mkdir(parents=True, exist_ok=True)
        else:
            results_dir = None

        if which == 'Garbage':
            train_set_neg, _ = data.get_train_set(include_negatives=True, has_background_class=True)
            test_set_all, test_set_neg, test_set_unkn = data.get_test_set(has_background_class=True)
        else:
            train_set_neg, _ = data.get_train_set(include_negatives=True, has_background_class=False)
            test_set_all, test_set_neg, test_set_unkn = data.get_test_set(has_background_class=False)

        if args.dataset == 'SmallScale':
            num_classes = 10
        else:
            if which != 'Garbage':
                num_classes = train_set_neg.label_count - 1
            else:
                num_classes = train_set_neg.label_count
        
        net = evals.load_network(args, which, num_classes)

        if net is None:
            print('net is none')
            continue

        #################################################################
        print('----- Prediction')
        #################################################################
        pred_results = {'train':None, 'test_neg':None, 'test_unkn':None, 'test_all':None}
        print("Train Extract")
        pred_results['train'] = evals.extract(train_set_neg, net, args.batch_size)
        print("Test Neg Extract")
        pred_results['test_neg'] = evals.extract(test_set_neg, net,  args.batch_size)
        print("Test Unknown Extract")
        pred_results['test_unkn'] = evals.extract(test_set_unkn, net,  args.batch_size)

        # Calculate Probs
        if which == "MultiBinary":
            train_probs = F.sigmoid(torch.tensor(pred_results['train'][1])).detach().numpy()
            test_neg_probs = F.sigmoid(torch.tensor(pred_results['test_neg'][1])).detach().numpy()
            test_unkn_probs  = F.sigmoid(torch.tensor(pred_results['test_unkn'][1])).detach().numpy()
        else:
            train_probs = F.softmax(torch.tensor(pred_results['train'][1]), dim=1).detach().numpy()
            test_neg_probs = F.softmax(torch.tensor(pred_results['test_neg'][1]), dim=1).detach().numpy()
            test_unkn_probs  = F.softmax(torch.tensor(pred_results['test_unkn'][1]), dim=1).detach().numpy()

        # remove the labels for the unknown class in case of Garbage Class
        if which == "Garbage":
            test_neg_probs = test_neg_probs[:,:-1]
            test_unkn_probs = test_unkn_probs[:,:-1]
            unkn_gt_label = 10  # Change the lable of unkn gt
        else:
            unkn_gt_label = -1

        pred_results['train'].append(train_probs)
        pred_results['test_neg'].append(test_neg_probs)
        pred_results['test_unkn'].append(test_unkn_probs)

        if args.pred_save:
            evals.eval_pred_save(pred_results, results_dir.joinpath('pred'))

        if args.arch == 'LeNet_plus_plus':
            #################################################################
            print('----- Deep Feature Plotting')
            #################################################################
            deep_features_plot(which, net, 
                               results_dir=results_dir, 
                               unkn_gt_label=unkn_gt_label, 
                               pred_results=pred_results, 
                               get_probs_fn=tools.viz.get_probs)


        #################################################################
        print('----- Get OSCR')
        #################################################################
        print("Test set : Positives + Negatives")
        ccr, fpr_neg = evals.get_oscr_curve(pred_results['test_neg'][0], pred_results['test_neg'][3]
                                            , unkn_gt_label, at_fpr=None)
        print("Test set : Positives + Unknowns")
        _, fpr_unkn = evals.get_oscr_curve(pred_results['test_unkn'][0], pred_results['test_unkn'][3]
                                           , unkn_gt_label,  at_fpr=None)
        print()
        results[which] = (ccr, fpr_neg, fpr_unkn)

    ################################################
    print('----- Plot OSCR')
    ################################################
    try:
        # plot with known unknowns (letters 1:13)
        pyplot.figure(figsize=(10,5))
        for which, res in results.items():
            pyplot.semilogx(res[1], res[0], label=labels[which])
        pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("Correct Classification Rate")
        pyplot.title("Negative Set")
        pyplot.tight_layout()
        pyplot.savefig(root.joinpath('oscr_neg.png'), bbox_inches="tight") 
        # pdf.savefig(bbox_inches='tight', pad_inches=0)
        
        # plot with unknown unknowns (letters 14:26)
        pyplot.figure(figsize=(10,5))
        for which, res in results.items():
            pyplot.semilogx(res[2], res[0], label=labels[which])
        pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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

