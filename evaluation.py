import time 
import pathlib

import torch
from torch.nn import functional as F

from library import tools, evals, dataset

def evaluate(args, config, seed):

    tools.set_seeds(seed)

    # load dataset
    if args.scale == 'SmallScale':
        data = dataset.EMNIST(config.data.smallscale.root,
                              split_ratio = 0.8, seed = seed,)
    else:
        data = dataset.IMAGENET(config.data.largescale.root,
                                protocol_root = config.data.largescale.protocol, 
                                protocol = int(args.scale.split('_')[1]), is_verbose=True)

    # Save or Plot results
    root = pathlib.Path(f"{args.scale}/_s{seed}/eval_{args.arch}")
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
        _, val_set_neg, num_classes = data.get_train_set()
        _, test_set_neg, test_set_unkn = data.get_test_set()
        unkn_gt_label = -1
        
        # Load weights of the model
        net = evals.load_network(args, config, which, num_classes, seed = seed) # 
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

    args = tools.eval_command_line_options()
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
        f"Seed: {args.seed}\n"
          )

    for s in args.seed:
        evaluate(args, config, s)
        print("\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\n\nEvaluation Done!")