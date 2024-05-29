import torch
import numpy
import os

from library import architectures, tools, losses, dataset

import matplotlib
matplotlib.rcParams["font.size"] = 18
from matplotlib import pyplot, patches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

########################################################################
# Reference Code
# 
# Author: Manuel Günther
# Date: 2024
# Availability: https://gitlab.uzh.ch/manuel.guenther/eos-example
########################################################################

labels={
  "SoftMax" : "Plain SoftMax",
  "Garbage" : "Garbage Class",
  "EOS" : "Entropic Open-Set",
  "Objectosphere" : "Objectosphere",
  "MultipleBinary" : "Multiple Binary Classifiers",
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
    parser.add_argument("--arch", default='LeNet_plus_plus', choices=['LeNet', 'LeNet_plus_plus'])
    parser.add_argument("--dataset_root", "-d", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--plot", "-p", default="Evaluate.pdf", help = "Where to write results into")
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()



def load_network(args,which):
    network_file = f"{args.arch}/{which}/{which}.model"
    if os.path.exists(network_file):
        net = architectures.__dict__[args.arch](use_BG=which=="Garbage",final_layer_bias=False) # NOTE: What is the effect of bias terms in the final layer? For both trianing and evaluation, we set it as False.
        net.load_state_dict(torch.load(network_file))
        tools.device(net)
        return net
    else:
        return None

def extract(dataset, net):
    gt, logits = [], []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

    with torch.no_grad():
        for (x, y) in data_loader:
            gt.extend(y.tolist())
            logs, feat = net(tools.device(x))
            logits.extend(logs.tolist())

    return numpy.array(gt), numpy.array(logits)

def find_ccr_at_fpr(FPR:numpy.array, CCR:numpy.array, ref_fpr:float):
    f = interp1d( FPR, CCR )
    ccr = f(ref_fpr).item()
    return ccr

def evaluate(args):

    # networks
    networks = {
        which: load_network(args, which) for which in args.approaches
    }

    emnist = dataset.EMNIST(args.dataset_root)

    results = {}
    for which, net in networks.items():
        if net is None:
            continue
        print ("Evaluating", which)

        # load test dataset
        _, val_set, test_set = emnist.get_test_set(has_background_class= which == "Garbage")

        # extract positives
        val_gt, val_predicted = extract(val_set, net)
        test_gt, test_predicted = extract(test_set, net)

        # compute probabilities
        val_predicted = torch.nn.functional.softmax(torch.tensor(val_predicted), dim=1).detach().numpy()
        test_predicted  = torch.nn.functional.softmax(torch.tensor(test_predicted ), dim=1).detach().numpy()

        unkn_gt_label = -1
        if which == "Garbage":
            # remove the labels for the unknown class in case of Garbage Class
            val_predicted = val_predicted[:,:-1]
            test_predicted = test_predicted[:,:-1]

            # Change the lable of unkn gt
            unkn_gt_label = 10 

        # vary thresholds
        ccr, fprv, fprt = [], [], []
        positives = val_predicted[val_gt != unkn_gt_label]
        val = val_predicted[val_gt == unkn_gt_label]
        test = test_predicted[test_gt == unkn_gt_label]
        gt = val_gt[val_gt != unkn_gt_label]
        for tau in sorted(positives[range(len(gt)),gt]):
            # correct classification rate
            ccr.append(numpy.sum(numpy.logical_and(
                numpy.argmax(positives, axis=1) == gt,
                positives[range(len(gt)),gt] >= tau
            )) / len(positives))
            # false positive rate for validation and test set
            fprv.append(numpy.sum(numpy.max(val, axis=1) >= tau) / len(val))
            fprt.append(numpy.sum(numpy.max(test, axis=1) >= tau) / len(test))

        ref_fpr = 0.01
        print(f"kn-unkn CCR@FPR{ref_fpr} : {find_ccr_at_fpr(numpy.array(fprv),numpy.array(ccr),ref_fpr)}")
        print(f"unkn-unkn CCR@FPR{ref_fpr} : {find_ccr_at_fpr(numpy.array(fprt),numpy.array(ccr),ref_fpr)}")

        results[which] = (ccr, fprv, fprt)

    pdf = PdfPages(args.plot)

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
        pdf.savefig(bbox_inches='tight', pad_inches=0)

        # plot with unknown unknowns (letters 14:26)
        pyplot.figure()
        for which, res in results.items():
            pyplot.semilogx(res[2], res[0], label=labels[which])
        pyplot.legend()
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("Correct Classification Rate")
        pyplot.title("Unknown Set")
        pyplot.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)

    finally:
        print("Wrote", args.plot)
        pdf.close()

if __name__ == '__main__':

    args = command_line_options()
    if args.gpu is not None and torch.cuda.is_available():
        tools.set_device_gpu(args.gpu)
    else:
        print("Running in CPU mode, might be slow")
        tools.set_device_cpu()

    evaluate(args)

