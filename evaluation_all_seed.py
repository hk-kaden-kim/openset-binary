import torch
from torch.nn import functional as F
import numpy
import os
import time 

from library import architectures, tools, evals, dataset, losses

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
  "EOS" : "Entropic Open-Set",
  "OvR" : "One-vs-Rest Classifiers",
  "OpenSetOvR": "Open-Set OvR Classifiers",
  "etc": None
}

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='...TBD'
    )

    parser.add_argument("--config", "-cf", default='config/eval.yaml', help="The configuration file that defines the experiment")
    parser.add_argument("--scale", "-sc", required=True, choices=['SmallScale', 'LargeScale', 'LargeScale_1', 'LargeScale_2', 'LargeScale_3'], help="Choose the scale of evaluation dataset.")
    parser.add_argument("--category", "-ct", required=True, choices=['_RQ1','_RQ2','_RQ3','_Discussion','_Tuning','etc'])
    parser.add_argument("--arch", "-ar", required=True)
    parser.add_argument("--approach", "-ap", nargs="+", default=list(labels.keys()), choices=list(labels.keys()), help = "Select the approaches to evaluate; non-existing models will automatically be skipped")
    parser.add_argument("--seed", "-s", default=42, nargs="+", type=int)
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()

def evaluate(args, config, seed):

    tools.set_seeds(seed)

    # load dataset
    if args.scale == 'SmallScale':
        data = dataset.EMNIST(config.data.smallscale.root,
                              split_ratio = 0.8, seed = seed,
                              label_filter=config.data.smallscale.label_filter,)
    else:
        data = dataset.IMAGENET(config.data.largescale.root,
                                protocol_root = config.data.largescale.protocol, 
                                protocol = int(args.scale.split('_')[1]), is_verbose=True)

    # Save or Plot results
    results = {}
    root = pathlib.Path(f"{args.scale}/_s{seed}/{args.category}/eval_{args.arch}")
    root.mkdir(parents=True, exist_ok=True)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(
        f"Configuration Details \n"
        f"Model Root: {config.arch.model_root}\n"
        f"Save Predictions: {config.pred_save==1}\n"
        f"Save OSCR results: {config.openset_save==1}\n"
          )

    for which in args.approach:

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print (f"Evaluation: {which}\n"
               f"Execution Time: {time.strftime('%d %b %Y %H:%M:%S')}\n")

        # Set the variables
        results_dir = root.joinpath(which)
        results_dir.mkdir(parents=True, exist_ok=True)

        pred_results = {'val':None, 'test_neg':None, 'test_unkn':None, 'test_all':None}

        if args.scale == 'SmallScale':
            batch_size = config.batch_size.smallscale
        else:
            batch_size = config.batch_size.largescale
        print(f"Batch Size: {batch_size}\n"
              f"Results: {results_dir}\n")

        # Load evaluation dataset
        _, val_set_neg, num_classes = data.get_train_set(is_verbose=True, size_train_negatives=config.data.train_neg_size, has_background_class=False)
        _, test_set_neg, test_set_unkn = data.get_test_set(is_verbose=True, has_background_class=False)
        unkn_gt_label = -1
        
        # Load weights of the model
        net = evals.load_network(args, config, which, num_classes, is_osovr=which=='OpenSetOvR', seed = seed) # 
        if net is None:
            print(f"Weights are not loaded on the network!\n{which} Evaluation Terminated\n")
            continue
        tools.device(net)

        # results [gt, logits, features]
        print("Execute predictions!")
        print(f"{time.strftime('%H:%M:%S')} Validation Set with 'Negative samples'...")
        pred_results['val'] = evals.extract(val_set_neg, net, batch_size, is_verbose=True)
        print(f"{time.strftime('%H:%M:%S')} Done!")
        print(f"{time.strftime('%H:%M:%S')} Test Set with 'Negative Samples'...")
        pred_results['test_neg'] = evals.extract(test_set_neg, net,  batch_size, is_verbose=True)
        print(f"{time.strftime('%H:%M:%S')} Done!")
        print(f"{time.strftime('%H:%M:%S')} Test Set with 'Unknown Samples'...")
        pred_results['test_unkn'] = evals.extract(test_set_unkn, net, batch_size, is_verbose=True)
        print(f"{time.strftime('%H:%M:%S')} Done!")
        print()

        # Calculate Probs
        if which == "OvR":
            # if args.scale == 'SmallScale':
            val_probs = F.sigmoid(torch.tensor(pred_results['val'][1])).detach().numpy()
            test_neg_probs = F.sigmoid(torch.tensor(pred_results['test_neg'][1])).detach().numpy()
            test_unkn_probs  = F.sigmoid(torch.tensor(pred_results['test_unkn'][1])).detach().numpy()
        
        elif which == 'OpenSetOvR':
            osovr_act = losses.OpenSetOvR(config.osovr_sigma.dict()[net.__class__.__name__])
            # if args.scale == 'SmallScale':
            val_probs = osovr_act(tools.device(torch.tensor(pred_results['val'][1]))).detach().cpu().numpy()
            test_neg_probs = osovr_act(tools.device(torch.tensor(pred_results['test_neg'][1]))).detach().cpu().numpy()
            test_unkn_probs  = osovr_act(tools.device(torch.tensor(pred_results['test_unkn'][1]))).detach().cpu().numpy()
        
        else:
            # if args.scale == 'SmallScale':
            val_probs = F.softmax(torch.tensor(pred_results['val'][1]), dim=1).detach().numpy()
            test_neg_probs = F.softmax(torch.tensor(pred_results['test_neg'][1]), dim=1).detach().numpy()
            test_unkn_probs  = F.softmax(torch.tensor(pred_results['test_unkn'][1]), dim=1).detach().numpy()
        
        # if args.scale == 'SmallScale':
        pred_results['val'].append(val_probs)
        pred_results['test_neg'].append(test_neg_probs)
        pred_results['test_unkn'].append(test_unkn_probs)

        if config.pred_save:
            evals.save_eval_pred(pred_results, results_dir.joinpath('pred'), save_feats = 'LeNet_plus_plus' in args.arch)

        print('Get Open-set evaluation results')
        print("1. Validation Set with 'Known Unknown Samples'...")
        ccr, fpr, urr, osa, thrs = evals.get_openset_perf(pred_results['val'][0], pred_results['val'][3], unkn_gt_label, is_verbose=True)
        if config.openset_save:
            evals.save_openset_perf('val', ccr, thrs, fpr, urr, osa, results_dir.joinpath('openset'))

        print("2. Test Set with 'Known Unknown Samples'...")
        ccr, fpr, urr, osa, thrs = evals.get_openset_perf(pred_results['test_neg'][0], pred_results['test_neg'][3], unkn_gt_label, is_verbose=True)
        if config.openset_save:
            evals.save_openset_perf('test_neg', ccr, thrs, fpr, urr, osa, results_dir.joinpath('openset'))

        print("3. Test Set with 'Unknown Unknown Samples'...")
        ccr, fpr, urr, osa, thrs = evals.get_openset_perf(pred_results['test_unkn'][0], pred_results['test_unkn'][3], unkn_gt_label, is_verbose=True)
        if config.openset_save:
            evals.save_openset_perf('test_unkn', ccr, thrs, fpr, urr, osa, results_dir.joinpath('openset'))

        print('Done!\n')
        torch.cuda.empty_cache()
        print('Release Unoccupied cache in GPU!')

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
        f"Category: {args.category} \n"
        f"Architecture: {args.arch} \n"
        f"Approach: {args.approach} \n"
        f"Configuration: {args.config} \n"
        f"Seed: {args.seed}\n"
        f"---------\nOSOvR Sigma: {config.osovr_sigma.dict()}\n"
          )

    for s in args.seed:
        evaluate(args, config, s)
        print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\n\nEvaluation Done!")
    

    #############################################
    # experiments = [('ResNet_50_C_b',['OvR']),
    #                ('ResNet_50_F_2',['OvR','OpenSetOvR']),
    #                ('ResNet_50_M_4',['OpenSetOvR']),
    #                ('ResNet_50_M_6',['OvR']),]
    # experiments = [('LeNet_plus_plus_F2C2',[0,1],0,2),
    #                ('LeNet_plus_plus_F2C2_neg_All',[0,1],-1,2),
    #                ('LeNet_plus_plus_F2C5',[0,1,2,3,4],0,2),
    #                ('LeNet_plus_plus_F2C5_neg_All',[0,1,2,3,4],-1,2),
    #                ('LeNet_plus_plus_F2C10',[-1],0,2),
    #                ('LeNet_plus_plus_F2C10_neg_All',[-1],-1,2),
    #                ('LeNet_plus_plus_F3C3',[0,1,2],0,3),
    #                ('LeNet_plus_plus_F3C3_neg_All',[0,1,2],-1,3),
    #                ('LeNet_plus_plus_F3C4',[0,1,2,3],0,3),
    #                ('LeNet_plus_plus_F3C4_neg_All',[0,1,2,3],-1,3)]
    # experiments = [('LeNet_C_neg_30k_b', ['OpenSetOvR'],  30000),
    #                ('LeNet_C_neg_30k_b', ['OvR'],         30000),

    #                ('LeNet_F_neg_30k_2', ['OpenSetOvR'],  30000),
    #                ('LeNet_F_neg_30k_3', ['OvR'],         30000),
                   
    #                ('LeNet_M_neg_30k_04', ['OpenSetOvR'], 30000),
    #                ('LeNet_M_neg_30k_02', ['OvR'],        30000),]
    
    # experiments = [['ResNet_50_C_neg_0_g',['OpenSetOvR','OvR'],'_RQ2'],
    #                ['ResNet_50_F_neg_0_1',['OvR'],'_RQ2'],
    #                ['ResNet_50_M_neg_0_04',['OpenSetOvR','OvR'],'_RQ2'],
    #                ['ResNet_50_C_neg_All_b',['OpenSetOvR'],'_RQ3'],
    #                ['ResNet_50_F_neg_All_1',['OpenSetOvR'],'_RQ3'],
    #                ['ResNet_50_M_neg_All_06',['OpenSetOvR'],'_RQ3'],
    #                ['ResNet_50_C_neg_All_g',['OvR'],'_RQ3'],
    #                ['ResNet_50_F_neg_All_3',['OvR'],'_RQ3'],]
    # print(experiments)
    # for s in args.seed:
    #     for item in experiments:
    #         args.arch = item[0]
    #         args.approach = item[1]
    #         args.category = item[2]
    #         print('Change to : ', args.arch, args.approach, args.category)
    #         evaluate(args, config, s)
    #         print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #         print("\n\nEvaluation Done!")
