import os
import torch
import numpy
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, balanced_accuracy_score, auc, f1_score, precision_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer

from ..architectures import architectures
from ..tools import device, set_device_cpu, get_device, print_table
from ..losses import confidence

# set_device_cpu()

########################################################################
# Reference Code
# 
# Author: Manuel Günther
# Date: 2024
# Availability: https://gitlab.uzh.ch/manuel.guenther/eos-example
########################################################################
class eval_results():
    def __init__(self, folder_path, has_train = False):
        
        try:
            if has_train:
                self.train_gt = numpy.load(os.path.join(folder_path, 'pred', 'train_gt.npy'))
                self.train_feats = numpy.load(os.path.join(folder_path, 'pred', 'train_feats.npy'))
                self.train_logits = numpy.load(os.path.join(folder_path, 'pred', 'train_logits.npy'))

            self.test_neg_gt = numpy.load(os.path.join(folder_path, 'pred', 'test_neg_gt.npy'))
            self.test_neg_logits = numpy.load(os.path.join(folder_path, 'pred', 'test_neg_logits.npy'))
            self.test_neg_probs = numpy.load(os.path.join(folder_path, 'pred', 'test_neg_probs.npy'))

            self.test_unkn_gt = numpy.load(os.path.join(folder_path, 'pred', 'test_unkn_gt.npy'))
            self.test_unkn_logits = numpy.load(os.path.join(folder_path, 'pred', 'test_unkn_logits.npy'))
            self.test_unkn_probs = numpy.load(os.path.join(folder_path, 'pred', 'test_unkn_probs.npy'))

            self.ccr = numpy.load(os.path.join(folder_path, 'openset', 'ccr.npy'))
            self.threshold = numpy.load(os.path.join(folder_path, 'openset', 'threshold.npy'))
            self.fpr_neg = numpy.load(os.path.join(folder_path, 'openset', 'fpr_neg.npy'))
            self.fpr_unkn = numpy.load(os.path.join(folder_path, 'openset', 'fpr_unkn.npy'))
            self.urr_neg = numpy.load(os.path.join(folder_path, 'openset', 'urr_neg.npy'))
            self.urr_unkn = numpy.load(os.path.join(folder_path, 'openset', 'urr_unkn.npy'))
            self.osa_neg = numpy.load(os.path.join(folder_path, 'openset', 'osa_neg.npy'))
            self.osa_unkn = numpy.load(os.path.join(folder_path, 'openset', 'osa_unkn.npy'))

        except Exception as error:

            if has_train:
                self.train_gt, self.train_feats, self.train_logits = None, None, None

            self.test_neg_gt, self.test_neg_logits, self.test_neg_probs = None, None, None
            self.test_unkn_gt, self.test_unkn_logits, self.test_unkn_probs = None, None, None
            
            self.ccr, self.threshold = None, None
            self.fpr_neg, self.fpr_unkn = None, None
            self.urr_neg, self.urr_unkn = None, None
            self.osa_neg, self.osa_unkn = None, None

            print(f"Error: Load evaluation results! {error}")

def plot_OSAC(data_info, lim=None, is_verbose=False):

    for d_i in data_info:

        info = d_i['info']
        
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        eval_res = eval_results(root_path)

        urr_unkn = eval_res.urr_unkn
        osa_neg = eval_res.osa_neg
        osa_unkn = eval_res.osa_unkn

        max_osa_neg_idx = numpy.argmax(osa_neg)
        max_osa_unkn_idx = numpy.argmax(osa_unkn)

        oprt_osa, oper_urr = osa_unkn[max_osa_neg_idx], urr_unkn[max_osa_neg_idx]
        orcl_osa, orcl_urr = osa_unkn[max_osa_unkn_idx], urr_unkn[max_osa_unkn_idx]

        plt.plot(urr_unkn, osa_unkn, label=d_i['label'])
        plt.scatter(oper_urr,oprt_osa,marker='*',facecolors='black',edgecolors='black',s=50, zorder=20)
        plt.scatter(orcl_urr,orcl_osa,marker='d',facecolors='none',edgecolors='black', zorder=20)
        if is_verbose:
            print(f"{d_i['label']} OSA (Operational / Oracle) = {oprt_osa:.4f} / {orcl_osa:.4f} ({oper_urr:.4f} / {orcl_urr:.4f})")

    if lim != None:
        plt.xlim(lim[0])
        plt.ylim(lim[1])
    else:
        plt.xlim((-0.02,1.02))
    plt.xlabel('Unknown Rejection Rate')
    plt.ylabel('Open-Set Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

def print_metrics(data_info):
    
    res = dict()

    print("acc\tprec\tf1score\tauroc-c\tauprc-o\tauroc-o\topenauc\toosa")
    for idx, d_i in enumerate(data_info):

        info = d_i['info']
        
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        eval_res = eval_results(root_path)

        max_score = numpy.max(eval_res.test_unkn_probs, axis=1)
        knowns = eval_res.test_unkn_gt != -1

        # Closed-set evaluation metrics
        known_gt = eval_res.test_unkn_gt[knowns]
        known_probs = eval_res.test_unkn_probs[knowns]
        known_pred = numpy.argmax(known_probs, axis=1)
        acc = compute_acc(known_gt, known_pred)
        precision = compute_precision(known_gt, known_pred,'macro')
        f1 = compute_f1score(known_gt, known_pred,'macro')
        auroc_c = compute_auroc(known_gt, known_probs, 'macro', by_class=True)

        # Open-set evaluation metrics
        auprc_o = compute_auprc(eval_res.test_unkn_gt, eval_res.test_unkn_probs, 'macro')
        auroc_o = compute_auroc(knowns, eval_res.test_unkn_probs, 'macro')
        openauc = compute_openauc(max_score[knowns], max_score[~knowns], known_pred, known_gt)
        oosa = compute_oosa(eval_res.osa_neg, eval_res.osa_unkn)
        print(f"{acc:.4f}\t{precision:.4f}\t{f1:.4f}\t{auroc_c:.4f}\t{auprc_o:.4}\t{auroc_o:.4f}\t{openauc:.4f}\t{oosa:.4f}")

        if idx == 0:
            res['acc'] = [acc]
            res['precision'] = [precision]
            res['f1'] = [f1]
            res['auroc_c'] = [auroc_c]
            res['auprc_o'] = [auprc_o]
            res['auroc_o'] = [auroc_o]
            res['openauc'] = [openauc]
            res['oosa'] = [oosa]
        else:
            res['acc'].append(acc)
            res['precision'].append(precision)
            res['f1'].append(f1)
            res['auroc_c'].append(auroc_c)
            res['auprc_o'].append(auprc_o)
            res['auroc_o'].append(auroc_o)
            res['openauc'].append(openauc)
            res['oosa'].append(oosa)

    return res

def compute_acc(known_gt, known_pred):
    acc = accuracy_score(known_gt, known_pred)
    return acc

def compute_bacc(known_gt, known_pred):
    bacc = balanced_accuracy_score(known_gt, known_pred)
    return bacc

def compute_precision(known_gt, known_pred, average):
    precision = precision_score(known_gt, known_pred, average=average)
    return precision

def compute_f1score(known_gt, known_pred, average):
    f1 = f1_score(known_gt, known_pred, average=average)
    return f1

def compute_multi_macro_auroc(gt, probs):

    fpr, tpr, roc_auc = dict(), dict(), 0

    n_classes = probs.shape[1]
    label_binarizer = LabelBinarizer().fit(gt)
    y_onehot_test = label_binarizer.transform(gt)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], probs[:, i])
        roc_auc += auc(fpr[i], tpr[i])

    return roc_auc / n_classes

def compute_auroc(gt, probs, average='macro', by_class=False):
    if average=='macro' and by_class:
        auroc = compute_multi_macro_auroc(gt, probs)
        # auroc = roc_auc_score(gt, probs, multi_class='ovr')
        return auroc
    max_probs = numpy.max(probs, axis=1)
    auroc = roc_auc_score(gt, max_probs, average=average)
    return auroc

def compute_openauc(max_prob_known, max_prob_unknown, pred, labels):
    """
    Reference : https://github.com/wang22ti/OpenAUC
    "OpenAUC: Towards AUC-Oriented Open-Set Recognition", Zitai Wang et al, NeurIPS 2022.

    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """
    # Open-set score function; if the score is larger than threshold, it is recognized as an open-set.
    x1 = - max_prob_known
    x2 = - max_prob_unknown

    x1, x2, correct = x1.tolist(), x2.tolist(), (pred == labels).tolist()
    m_x2 = max(x2) + 1e-5
    y_score = [value if hit else m_x2 for value, hit in zip(x1, correct)] + x2
    y_true = [0] * len(x1) + [1] * len(x2)
    openauc = roc_auc_score(y_true, y_score)

    return openauc

def compute_oosa(osa_neg, osa_unkn):
    oosa = osa_unkn[numpy.argmax(osa_neg)]
    return oosa

def compute_auprc(gt, probs, average):

    one_hot_gt = numpy.zeros(probs.shape)
    for i, c in enumerate(gt):
        if c != -1:
            one_hot_gt[i,c] = 1

    return average_precision_score(one_hot_gt, probs, average=average)

def plot_binary_OSAC(data_info, CMAP, lim=None):

    for i, d_i in enumerate(data_info):
        info = d_i['info']
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'

        results = eval_results(root_path)

        test_gt = results.test_neg_gt
        test_probs = results.test_neg_probs
        for c in range(10):
            targets = test_gt == c
            knowns = test_gt != -1

            bi_test_gt = numpy.zeros(test_gt.shape)
            bi_test_gt[~knowns] = -1
            bi_test_gt[numpy.logical_and(knowns, ~targets)] = 0
            bi_test_gt[targets] = 1

            bi_test_probs = test_probs[range(len(test_gt)), c]

            ccr, fpr = [], []
            for thr in sorted(bi_test_probs[~knowns]):
                ccr.append(numpy.sum(numpy.logical_or(numpy.logical_and(bi_test_gt == 1, bi_test_probs >= thr), 
                                                numpy.logical_and(bi_test_gt == 0, bi_test_probs < thr)))
                            / len(bi_test_probs))
                fpr.append(numpy.sum(numpy.logical_and(bi_test_gt == -1, bi_test_probs >= thr)) 
                        / len(bi_test_probs))


            # Get URR and OSA
            alpha = sum(bi_test_gt != -1) / len(bi_test_gt)
            urr = [1-v for v in fpr]
            osa = [alpha * c + (1-alpha) * u for c,u in zip(ccr,urr)]

            if c == 0:
                plt.plot(urr, osa, color=CMAP[i], alpha = 0.5, label=d_i['label'])
            else:
                plt.plot(urr, osa, color=CMAP[i], alpha = 0.5)

    if lim != None:
        plt.xlim(lim[0])
        plt.ylim(lim[1])
    else:
        plt.xlim((-0.02,1.02))
    plt.xlabel('Unknown Rejection Rate')
    plt.ylabel('Open-Set Accuracy')
    plt.grid(True)
    plt.legend()


def CCR_at_FPR(CCR, FPR, fpr_values=[1e-3,1e-2,1e-1,1]):
    """Computes CCR values for the desired FPR values, if such FPR values can be reached."""

    ccrs = []
    zero = numpy.zeros(FPR.shape)
    for desired_fpr in fpr_values:
        # get the FPR value that is closest, but above the current threshold
        candidates = numpy.nonzero(numpy.maximum(desired_fpr - FPR, zero))[0]
        if len(candidates) > 0:
            # there are values above threshold
            ccrs.append(CCR[candidates[0]])
        else:
            # the desired FPR cannot be reached
            ccrs.append(numpy.nan)

    return ccrs, fpr_values

def get_close_open_perf_results(data_info, seeds, FPR_vals=[1e-3,1e-2,1e-1,1]):
    results = []
    FPR_vals_all = numpy.linspace(FPR_vals[0],FPR_vals[-1],10**4)

    results = []
    for item in tqdm(data_info):

        info = item['info']
        conf_res_all, oscr_neg_res, oscr_unkn_res = [], [], []

        conf_res_all = []
        oscr_neg_res_all, oscr_neg_res = [], []
        oscr_unkn_res_all, oscr_unkn_res = [], []

        print(f"Computing {info}")
        for s in seeds:
            root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s{s}/{info[1]}/eval_{info[2]}/{info[3]}'
            eval_res = eval_results(root_path)
        
            ccrs_all, _ = CCR_at_FPR(eval_res.ccr, eval_res.fpr_neg, FPR_vals_all)
            ccrs, _ = CCR_at_FPR(eval_res.ccr, eval_res.fpr_neg, FPR_vals)
            oscr_neg_res_all.append(ccrs_all)
            oscr_neg_res.append(ccrs)

            ccrs_all, _ = CCR_at_FPR(eval_res.ccr, eval_res.fpr_unkn, FPR_vals_all)
            ccrs, _ = CCR_at_FPR(eval_res.ccr, eval_res.fpr_unkn, FPR_vals)
            oscr_unkn_res_all.append(ccrs_all)
            oscr_unkn_res.append(ccrs)
        
            conf_neg_res = confidence(torch.Tensor(eval_res.test_neg_probs), torch.Tensor(eval_res.test_neg_gt).type(dtype=torch.int),
                                            offset = 0., unknown_class = -1, last_valid_class = None,)
            conf_unkn_res = confidence(torch.Tensor(eval_res.test_unkn_probs), torch.Tensor(eval_res.test_unkn_gt).type(dtype=torch.int),
                                            offset = 0., unknown_class = -1, last_valid_class = None,)
            r_kn, r_kn_unkn = conf_neg_res[0]/conf_neg_res[1], conf_neg_res[2]/conf_neg_res[3]
            r_unkn_unkn = conf_unkn_res[2]/conf_unkn_res[3]

            r = (r_kn + r_kn_unkn + r_unkn_unkn) / 3
            conf_res_all.append([r.item(),r_kn.item(),r_kn_unkn.item(),r_unkn_unkn.item()])

        conf_res_all = numpy.array(conf_res_all)
        oscr_neg_res_all = numpy.array(oscr_neg_res_all)
        oscr_neg_res = numpy.array(oscr_neg_res)
        oscr_unkn_res_all = numpy.array(oscr_unkn_res_all)
        oscr_unkn_res = numpy.array(oscr_unkn_res)

        conf_res_all_avg = numpy.nanmean(conf_res_all,axis=0)
        oscr_neg_res_all_med = numpy.median(numpy.nan_to_num(oscr_neg_res_all, copy=False),axis=0)
        oscr_neg_res_med = numpy.median(numpy.nan_to_num(oscr_neg_res, copy=False),axis=0)
        oscr_neg_range = (oscr_neg_res_med-numpy.min(oscr_neg_res,axis=0), numpy.max(oscr_neg_res,axis=0)-oscr_neg_res_med)
        oscr_unkn_res_all_med = numpy.median(numpy.nan_to_num(oscr_unkn_res_all, copy=False),axis=0)
        oscr_unkn_res_med = numpy.median(numpy.nan_to_num(oscr_unkn_res, copy=False),axis=0)
        oscr_unkn_range = (oscr_unkn_res_med-numpy.min(oscr_unkn_res,axis=0), numpy.max(oscr_unkn_res,axis=0)-oscr_unkn_res_med)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            results.append([(conf_res_all_avg, 
                            oscr_neg_res_all_med, 
                            oscr_neg_res_med, 
                            oscr_neg_range),
                            (_, 
                            oscr_unkn_res_all_med, 
                            oscr_unkn_res_med, 
                            oscr_unkn_range)])
    return results

def print_conf_ccrfpr_acc(results, FPR_vals=[1e-3,1e-2,1e-1,1]):
    print("Confidence Score")
    print(*['r','r+','r+-','r--'], sep='\t')
    for idx, res in enumerate(results):
        print(*[f"{x:.4f}" for x in res[0][0]], sep='\t')
    print()

    print("CCR@FPR (Testset: Known + Negatives)")
    print(*FPR_vals[:-1], sep='\t')
    for idx, res in enumerate(results):
        print(*[f"{x:.4f}" if x!=0 else '-' for x in res[0][2][:-1]], sep='\t')
    print()

    print("CCR@FPR (Testset: Known + Unknowns)")
    print(*FPR_vals[:-1], sep='\t')
    for idx, res in enumerate(results):
        print(*[f"{x:.4f}" if x!=0 else '-' for x in res[1][2][:-1]], sep='\t')
    print()

    print("CCR@FPR of 1")
    for idx, res in enumerate(results):
        print(f"{res[1][2][-1]:.4f}")

def plot_oscr_all(results_plot, data_info_plot, CMAP, lines=None, FPR_vals=[1e-3,1e-2,1e-1,1], ylim=(-0.05,1.05)):

    if lines == None: lines = ['-']*len(data_info_plot)
    FPR_vals_all = numpy.linspace(FPR_vals[0],FPR_vals[-1],10**4)
    
    # plot with Negatives
    plt.figure(figsize=(6,3))
    for idx, res in enumerate(results_plot):
        plt.semilogx(FPR_vals_all, res[0][1], label=data_info_plot[idx]['label'], alpha=1, linestyle=lines[idx], color = CMAP[idx])
    plt.xlabel("FPR")
    plt.ylabel("CCR")
    plt.xlim((4e-4,2.5))
    plt.ylim(ylim)
    plt.title("Negative Set")
    plt.tight_layout()
    plt.tick_params(direction='in', which='both', top=True, right=True)
    plt.grid(which='major', linestyle=':')
    # plt.savefig(root.joinpath('oscr_neg.png'), bbox_inches="tight") 

    # plot with Unknowns
    plt.figure(figsize=(6,3))
    for idx, res in enumerate(results_plot):
        plt.semilogx(FPR_vals_all, res[1][1], label=data_info_plot[idx]['label'], alpha=1, linestyle=lines[idx], color = CMAP[idx])
    plt.xlabel("FPR")
    plt.ylabel("CCR")
    plt.xlim((4e-4,2.5))
    plt.ylim(ylim)
    plt.title("Unknown Set")
    plt.tight_layout()
    plt.tick_params(direction='in', which='both', top=True, right=True)
    plt.grid(which='major', linestyle=':')
    # plt.savefig(root.joinpath('oscr_unkn.png'), bbox_inches="tight") 


    # Plot legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figure(figsize=(3,2))
    plt.legend(handles=handles, loc='center',).axes.axis('off')

def plot_oscr_detail(results_plot, data_info_plot, CMAP, markers=None, FPR_vals=[1e-3,1e-2,1e-1,1], ylim=(-0.05,1.05)):
    
    if markers == None: markers = ['o']*len(data_info_plot)

    trans_r = 2.5 if len(results_plot) > 8 else 10
    trans = numpy.arange(len(results_plot)) * 0.1
    trans = trans_r ** (trans-numpy.mean(trans))

    fig, ax = plt.subplots(figsize=(6,3))
    for idx, res in enumerate(results_plot):
        er1 = ax.errorbar([v * trans[idx] for v in FPR_vals], res[0][2], yerr=res[0][3], linestyle="none", capsize=3, 
                        marker=markers[idx], label=data_info_plot[idx]['label'], color = CMAP[idx])

    plt.xlabel("FPR")
    plt.xscale('log')
    plt.ylabel("CCR")
    plt.xlim((4e-4,2.5))
    plt.ylim(ylim)
    plt.title("Negative Set")
    plt.tick_params(direction='in', which='both', top=True, right=True)
    plt.grid(which='major', linestyle=':')
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(6,3))
    for idx, res in enumerate(results_plot):
        er1 = ax.errorbar([v * trans[idx] for v in FPR_vals], res[1][2], yerr=res[1][3], linestyle="none", capsize=3,
                        marker=markers[idx], label=data_info_plot[idx]['label'], color = CMAP[idx])

    plt.xlabel("FPR")
    plt.xscale('log')
    plt.ylabel("CCR")
    plt.xlim((4e-4,2.5))
    plt.ylim(ylim)
    plt.title("Unknown Set")
    plt.tick_params(direction='in', which='both', top=True, right=True)
    plt.grid(which='major', linestyle=':')
    plt.tight_layout()

    # Plot legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figure(figsize=(3,2))
    plt.legend(handles=handles, loc='center',).axes.axis('off')

def plot_dist_prob(data_info, num_classes, bins, seeds, figsize=(10,3), ylim=(5e-7, 1.5)):

    center = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(1,len(data_info),figsize=figsize, sharey=True)
    
    for idx in tqdm(range(len(data_info))):
        # plt.figure(figsize=(6,3))
        info = data_info[idx]['info']

        target_probs_all = numpy.array([])
        non_target_probs_all = numpy.array([])
        negative_probs_all = numpy.array([])
        unknown_probs_all = numpy.array([])

        for c in range(num_classes):
            for s in seeds:

                # Load evaluation results
                root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s{s}/{info[1]}/eval_{info[2]}/{info[3]}'
                # folder_path = f"./_results/{item['info'][0]}/_s{s}/eval_{item['info'][1]}/{item['info'][2]}"
                results = eval_results(root_path)

                # Get known and unknown samples
                knowns = results.test_neg_gt != -1
                negatives = results.test_neg_gt == -1
                unknowns = results.test_unkn_gt == -1

                # Get target and non-target samples
                targets = numpy.logical_and(knowns, results.test_neg_gt == c)
                non_targets = numpy.logical_and(knowns, results.test_neg_gt != c)

                # Get probabilities for each item
                target_probs_all = numpy.append(target_probs_all, results.test_neg_probs[targets,:][:,c])
                non_target_probs_all = numpy.append(non_target_probs_all, results.test_neg_probs[non_targets,:][:,c])
                negative_probs_all = numpy.append(negative_probs_all, results.test_neg_probs[negatives,:][:,c])
                unknown_probs_all = numpy.append(unknown_probs_all, results.test_unkn_probs[unknowns,:][:,c])

        # Get histogram info
        target_probs_all_hist, _ = numpy.histogram(target_probs_all, bins=bins, density=False)
        non_target_probs_all_hist, _ = numpy.histogram(non_target_probs_all, bins=bins, density=False)
        negative_probs_all_hist, _ = numpy.histogram(negative_probs_all, bins=bins, density=False)
        unknown_probs_all_hist, _ = numpy.histogram(unknown_probs_all, bins=bins, density=False)

        # Change it to ratio between 0 and 1
        target_probs_ratio = target_probs_all_hist/sum(target_probs_all_hist)
        non_target_probs_ratio = non_target_probs_all_hist/sum(non_target_probs_all_hist)
        negative_probs_ratio = negative_probs_all_hist/sum(negative_probs_all_hist)
        unknown_probs_ratio = unknown_probs_all_hist/sum(unknown_probs_all_hist)

        ax[idx].plot(center, unknown_probs_ratio, color = 'black', label='Unknowns')
        ax[idx].plot(center, negative_probs_ratio, color = 'black', linestyle='-.', label='Negatives')
        ax[idx].plot(center, non_target_probs_ratio, color = 'red', label='Non-Target')
        ax[idx].plot(center, target_probs_ratio, color = 'green',label='Target')

        # plt.title('Target vs. Non-Target vs. Negatives')
        # ax[idx].set_xlabel('probability score')
        if idx == 0: ax[idx].set_ylabel('ratio')
        ax[idx].set_ylim(ylim)
        ax[idx].set_yscale('log')
        ax[idx].set_title(f"{data_info[idx]['label']}")
        ax[idx].grid(which='major', linestyle=':')
        
    fig.supxlabel('probability score')
    # fig.supylabel('ratio')
    fig.tight_layout()

    # Plot legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figure(figsize=(4,1))
    plt.legend(handles=handles, loc='center', ncol=len(handles)).axes.axis('off')
    # ax.legend(handles=legend_elements, bbox_to_anchor=(.55, -.15))
    # # fig.suptitle(f"{item['label']}")
    plt.tight_layout()




def load_network(args, config, which, num_classes, is_osovr=False, seed=-1):

    network_file = os.path.join(config.arch.model_root, f"{args.scale}/_s{seed}/{args.category}/{args.arch}/{which}")
    network_file = os.path.join(network_file, f"{which}.model")

    print(network_file)
    if os.path.exists(network_file):

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
        net = architectures.__dict__[arch_name](use_BG=False,
                                                num_classes=num_classes,
                                                final_layer_bias=False,
                                                feat_dim=config.arch.feat_dim,
                                                is_osovr=is_osovr)
        checkpoint = torch.load(network_file, map_location=torch.device('cpu')) 

        # if config.need_sync:
        #     print('Weights are came from the reference code! Sync the weight name!')
        #     checkpoint = architectures.checkpoint_sync(checkpoint["model_state_dict"], map_location=torch.device('cpu'))     
        
        net.load_state_dict(checkpoint)
        device(net)

        return net
    return None

def extract(dataset, net, batch_size=2048, is_verbose=False):
    '''
    return : gt, logits, feats
    '''
    gt, logits, feats = [], [], []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (x, y) in tqdm(data_loader, miniters=int(len(data_loader)/3), maxinterval=600, disable=not is_verbose):
            gt.extend(y.tolist())
            logs, feat = net(device(x))
            logits.extend(logs.tolist())
            feats.extend(feat.tolist())

    gt = numpy.array(gt)
    logits = numpy.array(logits)
    feats = numpy.array(feats)

    print("\nEvaluation Dataset Stats:")
    stats = numpy.unique(gt, return_counts=True)
    print_table(stats[0], stats[1])

    return [gt, logits, feats]

def find_ccr_at_fpr(FPR:numpy.array, CCR:numpy.array, ref_fpr:float):
    f = interp1d( FPR, CCR )
    ccr = f(ref_fpr).item()
    return ccr

def get_openset_perf(test_gt:numpy.array, test_probs:numpy.array, unkn_gt_label=-1, at_fpr=None, is_verbose=False):

    # vary thresholds
    ccr, fpr = [], []
    kn_probs = test_probs[test_gt != unkn_gt_label]
    unkn_probs = test_probs[test_gt == unkn_gt_label]
    gt = test_gt[test_gt != unkn_gt_label]

    # Get CCR and FPR
    sorted_kn_probs = sorted(kn_probs[range(len(gt)),gt])
    for tau in tqdm(sorted_kn_probs, miniters=int(len(gt)/5), maxinterval=600, disable=not is_verbose):
        # correct classification rate
        ccr.append(numpy.sum(numpy.logical_and(
            numpy.argmax(kn_probs, axis=1) == gt,
            kn_probs[range(len(gt)),gt] >= tau
        )) / len(kn_probs))
        # false positive rate for validation and test set
        fpr.append(numpy.sum(numpy.max(unkn_probs, axis=1) >= tau) / len(unkn_probs))
    
    # Get URR and OSA
    alpha = sum(test_gt != unkn_gt_label) / len(test_gt)
    urr = [1-v for v in fpr]
    osa = [alpha * c + (1-alpha) * u for c,u in zip(ccr,urr)]

    if at_fpr is not None:
        print(f"\tCCR@FPR{at_fpr} : {find_ccr_at_fpr(numpy.array(fpr),numpy.array(ccr),at_fpr)}")

    return (ccr, fpr, urr, osa, sorted_kn_probs)

def save_eval_pred(pred_results:dict, root:str, save_feats=True):

    if not os.path.exists(root):
        os.makedirs(root)
        print(f"Folder '{root}' created successfully.")

    # Save the dictionary keys
    keys_list = list(pred_results.keys())
    keys_array = numpy.array(keys_list)
    numpy.save(os.path.join(root,'keys.npy'), keys_array)

    # Save each NumPy array in the values
    pred_name = ['gt', 'logits', 'feats', 'probs']
    for key, value in pred_results.items():
        if value:
            for i, arr in enumerate(value):
                if not save_feats and i == 2:
                    continue
                numpy.save(os.path.join(root, f'{key}_{pred_name[i]}.npy'), arr)
    
    print(f"Prediction Saved Successfully!\n{root}\n")

def save_openset_perf(ccr:list, threshold:list, 
                      fpr_neg:list, fpr_unkn:list, 
                      urr_neg:list, urr_unkn:list, 
                      osa_neg:list, osa_unkn:list, 
                      root:str):

    if not os.path.exists(root):
        os.makedirs(root)
        print(f"Folder '{root}' created successfully.")

    numpy.save(os.path.join(root, 'ccr.npy'), ccr)
    numpy.save(os.path.join(root, 'threshold.npy'), threshold)
    numpy.save(os.path.join(root, 'fpr_neg.npy'), fpr_neg)
    numpy.save(os.path.join(root, 'fpr_unkn.npy'), fpr_unkn)
    numpy.save(os.path.join(root, 'urr_neg.npy'), urr_neg)
    numpy.save(os.path.join(root, 'urr_unkn.npy'), urr_unkn)
    numpy.save(os.path.join(root, 'osa_neg.npy'), osa_neg)
    numpy.save(os.path.join(root, 'osa_unkn.npy'), osa_unkn)

    print(f"Open-set performance Data Saved Successfully!\n{root}\n")





########################################################################
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

def common_processing(gt, predicted_class, score, knownness_score=None):
    """
    Returns samples sorted by knownness scores along with unique scores/thresholds
    :param gt:
    :param predicted_class:
    :param score:
    :return:
    """
    if len(predicted_class.shape) != 2:
        predicted_class = predicted_class[:, None]
    if len(score.shape) != 2:
        score = score[:, None]

    if knownness_score is None:
        if len(score.shape) != 1:
            knownness_score = torch.max(score, dim=1).values
        else:
            knownness_score = score.clone()
    knownness_score = device(knownness_score)

    # Sort samples in decreasing order of knownness
    knownness_score, indices = torch.sort(knownness_score, descending=True)
    indices = indices.cpu()
    predicted_class, gt, score = predicted_class[indices], gt[indices], score[indices]
    del indices

    # Perform score tie breaking
    # The last occurence of the highest threshold is to be preserved

    # sort knownness scores in an ascending order to find unique occurences
    scores_reversed = knownness_score[torch.arange(knownness_score.shape[0] - 1, -1, -1)]
    unique_scores_reversed, counts_reversed = torch.unique_consecutive(
        scores_reversed, return_counts=True
    )
    del scores_reversed
    # Reverse again to get scores & counts in descending order
    indx = torch.arange(unique_scores_reversed.shape[0] - 1, -1, -1)
    unique_scores, counts = unique_scores_reversed[indx], counts_reversed[indx]
    del unique_scores_reversed, counts_reversed

    threshold_indices = torch.cumsum(counts, dim=-1) - 1
    return gt, predicted_class, score, unique_scores, threshold_indices


def get_known_unknown_indx(gt, predicted_class, unknown_labels={-1}):
    # Get all indices for knowns and unknowns
    all_known_indexs = []
    for unknown_label in unknown_labels:
        all_known_indexs.append(gt != unknown_label)
    all_known_indexs = torch.stack(all_known_indexs)
    known_indexs = all_known_indexs.all(dim=0)
    unknown_indexs = ~known_indexs
    del all_known_indexs
    return known_indexs, unknown_indexs


def get_correct_for_accuracy(gt, predicted_class, score, topk=1):
    score, prediction_indx = torch.topk(score, topk, dim=1)
    prediction_made = torch.gather(predicted_class, 1, prediction_indx)
    correct_bool = torch.any(gt[:, None].cpu() == prediction_made, dim=1)
    correct_cumsum = torch.cumsum(correct_bool, dim=-1).type("torch.FloatTensor")
    return correct_bool, correct_cumsum


def tensor_OSRC(gt, predicted_class, score, knownness_score=None):
    gt, predicted_class, score, unique_scores, threshold_indices = common_processing(
        gt, predicted_class, score, knownness_score
    )
    gt = device(gt)

    known_indexs, unknown_indexs = get_known_unknown_indx(gt, predicted_class)

    # Get the denominators for accuracy and OSE
    no_of_knowns = known_indexs.sum().type("torch.FloatTensor")
    no_of_unknowns = unknown_indexs.sum().type("torch.FloatTensor")

    # any known samples seen for a score > \theta is a known unless the sample was predicted as an unknown, i.e. predicted_class == -1 \
    # Note: incase of topk if any of the top k predictions is -1 then it is considered unknown
    knowns_not_detected_as_unknowns = known_indexs
    knowns_not_detected_as_unknowns[torch.any(predicted_class == -1, dim=1)] = False
    current_converage = torch.cumsum(knowns_not_detected_as_unknowns, dim=0) / no_of_knowns

    correct_bool, _ = get_correct_for_accuracy(gt, predicted_class, score)
    # if a known was detected as unknown it should not be marked as correct
    correct_bool = correct_bool * knowns_not_detected_as_unknowns
    correct_cumsum = torch.cumsum(correct_bool, dim=-1).type("torch.FloatTensor")
    knowns_accuracy = correct_cumsum / no_of_knowns

    # any unknown sample seen for a score > \theta is a false positive unless the sample was predicted as an unknown, i.e. predicted_class == -1 \
    # Note: incase of topk if any of the top k predictions is -1 then it is considered unknown
    unknown_detected_as_known = unknown_indexs
    unknown_detected_as_known[torch.any(predicted_class == -1, dim=1)] = False
    all_FPs = torch.cumsum(unknown_detected_as_known, dim=-1).type("torch.FloatTensor")
    OSE = all_FPs / no_of_unknowns

    knowns_accuracy, current_converage, OSE = (
        knowns_accuracy[threshold_indices],
        current_converage[threshold_indices],
        OSE[threshold_indices],
    )
    return (OSE, knowns_accuracy, current_converage)


def coverage(gt, predicted_class, score, knownness_score=None):
    gt, predicted_class, score, unique_scores, threshold_indices = common_processing(
        gt, predicted_class, score, knownness_score
    )
    correct_bool, correct_cumsum = get_correct_for_accuracy(gt, predicted_class, score)
    acc = correct_cumsum[threshold_indices] / gt.shape[0]
    incorrect_cumsum = torch.cumsum(~correct_bool, dim=-1).type("torch.FloatTensor")
    incorrect_cumsum = incorrect_cumsum[threshold_indices]
    current_converage = (threshold_indices + 1) / gt.shape[0]
    return unique_scores, incorrect_cumsum, acc, current_converage


def calculate_binary_precision_recall(gt, predicted_class, score):
    """
                                Detected as
                            Knowns        Unknowns
    Ground Truth  Knowns      TP             FN
                Unknowns      FP             TN
    This function measures the performance of an algorithm to identify a known as a known while
    the unknowns only impact as false positives in the precision
    :param gt:
    :param predicted_class:
    :param score:
    :return:
    """
    gt, predicted_class, score, unique_scores, threshold_indices = common_processing(
        gt, predicted_class, score
    )
    known_indexs, unknown_indexs = get_known_unknown_indx(gt, predicted_class)

    no_of_knowns = known_indexs.sum().type("torch.FloatTensor")

    # any known samples seen for a score > \theta is a known unless the sample was predicted as an unknown, i.e. predicted_class == -1 \
    # Note: incase of topk if any of the top k predictions is -1 then it is considered unknown
    knowns_not_detected_as_unknowns = known_indexs
    knowns_not_detected_as_unknowns[torch.any(predicted_class == -1, dim=1)] = False
    all_knowns = torch.cumsum(knowns_not_detected_as_unknowns, dim=-1).type(
        "torch.FloatTensor"
    )

    # any unknown sample seen for a score > \theta is a false positive unless the sample was predicted as an unknown, i.e. predicted_class == -1 \
    # Note: incase of topk if any of the top k predictions is -1 then it is considered unknown
    unknown_detected_as_known = unknown_indexs
    unknown_detected_as_known[torch.any(predicted_class == -1, dim=1)] = False
    all_unknowns = torch.cumsum(unknown_detected_as_known, dim=-1).type(
        "torch.FloatTensor"
    )

    all_knowns, all_unknowns = (
        all_knowns[threshold_indices],
        all_unknowns[threshold_indices],
    )

    Recall = all_knowns / no_of_knowns
    # Precision here is non monotonic
    Precision = all_knowns / (all_knowns + all_unknowns)

    return Precision, Recall, unique_scores


def F_score(Precision, Recall, ß=1.0):
    """
    Calculates F Score by the following equation, default is F1 score because ß = 1.0
    F_Score = (1+ß**2)*((Precision * Recall) / ((ß**2)*Precision + Recall))
    :param Precision:
    :param Recall:
    :param ß:
    :return:
    """
    FScore = (1 + (ß ** 2)) * ((Precision * Recall) / (((ß ** 2) * Precision) + Recall))
    return FScore
