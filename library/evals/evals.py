import os
import torch
import numpy
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib import ticker
from matplotlib.lines import Line2D
import networkx as nx

import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, balanced_accuracy_score, auc, f1_score, precision_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelBinarizer

from ..architectures import architectures
from ..tools import device, set_device_cpu, get_device, print_table
from ..losses import confidence

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import warnings
warnings.filterwarnings("ignore")

# set_device_cpu()


def repel_labels(ax, label_data, k=0.01):
    G = nx.DiGraph()
    data_nodes = []
    init_pos = {}

    x = [d[0] for d in label_data]
    y = [d[1] for d in label_data]
    labels = [d[2] for d in label_data]

    for xi, yi, label in zip(x, y, labels):
        data_str = 'data_{0}'.format(label)
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

    # undo spring_layout's rescaling
    pos_after = numpy.vstack([pos[d] for d in data_nodes])
    pos_before = numpy.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = numpy.polyfit(pos_after[:,0], pos_before[:,0], 1)
    scale, shift_y = numpy.polyfit(pos_after[:,1], pos_before[:,1], 1)
    shift = numpy.array([shift_x, shift_y])
    for key, val in pos.items():
        pos[key] = (val*scale) + shift

    for label, data_str in G.edges():
        ax.annotate(label,
                    xy=pos[data_str], xycoords='data',
                    xytext=pos[label], textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3", 
                                    color='red'), )
    # expand limits
    all_pos = numpy.vstack([list(v) for v in pos.values()])
    x_span, y_span = numpy.ptp(all_pos, axis=0)
    mins = numpy.min(all_pos-x_span*0.15, 0)
    maxs = numpy.max(all_pos+y_span*0.15, 0)
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])

########################################################################
# Reference Code
# 
# Author: Manuel Günther
# Date: 2024
# Availability: https://gitlab.uzh.ch/manuel.guenther/eos-example
########################################################################
class eval_results():
    def __init__(self, folder_path, load_feats=False):
        
        try:
            # Prediction results
            self.val_gt = numpy.load(os.path.join(folder_path, 'pred', 'val_gt.npy'))
            self.val_logits = numpy.load(os.path.join(folder_path, 'pred', 'val_logits.npy'))
            self.val_probs = numpy.load(os.path.join(folder_path, 'pred', 'val_probs.npy'))
            if load_feats:
                self.val_feats = numpy.load(os.path.join(folder_path, 'pred', 'val_feats.npy'))

            self.test_neg_gt = numpy.load(os.path.join(folder_path, 'pred', 'test_neg_gt.npy'))
            self.test_neg_logits = numpy.load(os.path.join(folder_path, 'pred', 'test_neg_logits.npy'))
            self.test_neg_probs = numpy.load(os.path.join(folder_path, 'pred', 'test_neg_probs.npy'))

            self.test_unkn_gt = numpy.load(os.path.join(folder_path, 'pred', 'test_unkn_gt.npy'))
            self.test_unkn_logits = numpy.load(os.path.join(folder_path, 'pred', 'test_unkn_logits.npy'))
            self.test_unkn_probs = numpy.load(os.path.join(folder_path, 'pred', 'test_unkn_probs.npy'))
            
            # Performance results
            self.val_ccr = numpy.load(os.path.join(folder_path, 'openset', 'val_ccr.npy'))
            self.val_thrs = numpy.load(os.path.join(folder_path, 'openset', 'val_thrs.npy'))
            self.val_fpr = numpy.load(os.path.join(folder_path, 'openset', 'val_fpr.npy'))
            self.val_urr = numpy.load(os.path.join(folder_path, 'openset', 'val_urr.npy'))
            self.val_osa = numpy.load(os.path.join(folder_path, 'openset', 'val_osa.npy'))

            self.test_neg_ccr = numpy.load(os.path.join(folder_path, 'openset', 'test_neg_ccr.npy'))
            self.test_neg_thrs = numpy.load(os.path.join(folder_path, 'openset', 'test_neg_thrs.npy'))
            self.test_neg_fpr = numpy.load(os.path.join(folder_path, 'openset', 'test_neg_fpr.npy'))
            self.test_neg_urr = numpy.load(os.path.join(folder_path, 'openset', 'test_neg_urr.npy'))
            self.test_neg_osa = numpy.load(os.path.join(folder_path, 'openset', 'test_neg_osa.npy'))

            self.test_unkn_ccr = numpy.load(os.path.join(folder_path, 'openset', 'test_unkn_ccr.npy'))
            self.test_unkn_thrs = numpy.load(os.path.join(folder_path, 'openset', 'test_unkn_thrs.npy'))
            self.test_unkn_fpr = numpy.load(os.path.join(folder_path, 'openset', 'test_unkn_fpr.npy'))
            self.test_unkn_urr = numpy.load(os.path.join(folder_path, 'openset', 'test_unkn_urr.npy'))
            self.test_unkn_osa = numpy.load(os.path.join(folder_path, 'openset', 'test_unkn_osa.npy'))

        except Exception as error:

            self.val_gt, self.val_logits, self.val_probs = None, None, None
            if load_feats:
                self.val_feats = None

            self.test_neg_gt, self.test_neg_logits, self.test_neg_probs = None, None, None
            self.test_unkn_gt, self.test_unkn_logits, self.test_unkn_probs = None, None, None
            
            self.ccr, self.threshold = None, None
            self.fpr_neg, self.fpr_unkn = None, None
            self.urr_neg, self.urr_unkn = None, None
            self.osa_neg, self.osa_unkn = None, None

            # print(f"Error: Load evaluation results! {error}")

def plot_OSA(data_info, colors, labels=None, figsize=(5,3), lim=None, show_val=False, show_point=(True,True), 
              zoom=((False, (0.7,0.8,0.7,0.8),(0.1,0.1,0.1,0.1)), (False, (0.7,0.8,0.7,0.8),(0.1,0.1,0.1,0.1)))):

    if labels == None:
        labels = range(len(data_info))

    legend_item_oosa = Line2D([0], [0], color='black', marker='*', linestyle='None', markersize=7, markerfacecolor='none')
    legend_item_iosa = Line2D([0], [0], color='black', marker='d', linestyle='None', markersize=5, markerfacecolor='none')

    # Get validation set results and the operational threshold
    eval_res = []
    for idx, d_i in enumerate(data_info):

        info = d_i['info']
        
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        res = eval_results(root_path)
        if res.val_gt is None:
            continue
        else:
            urr = res.val_urr
            osa = res.val_osa
            thrs = res.val_thrs
            op_thrs = thrs[numpy.argmax(osa)]
        
        eval_res.append({'res':res, 'op_thrs':op_thrs})

    # Plot OOSA for the test set with negative samples
    # plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    if zoom[0][0]:
        x1, x2, y1, y2 = zoom[0][1]  # subregion of the original image
        axins = ax.inset_axes(
            zoom[0][2] ,
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    else:
        axins = None
        
    for idx, d_i in enumerate(data_info):
        if show_val:
            ax.plot(eval_res[idx]['res'].val_urr, eval_res[idx]['res'].val_osa, color=colors[idx], alpha=0.1, linewidth=5)

        urr = eval_res[idx]['res'].test_neg_urr
        osa = eval_res[idx]['res'].test_neg_osa
        thrs = eval_res[idx]['res'].test_neg_thrs

        op_idx = numpy.argmax(thrs > eval_res[idx]['op_thrs']) - 1
        op_osa, op_urr = osa[op_idx], urr[op_idx]
        id_idx = numpy.argmax(osa)
        id_osa, id_urr, id_thrs = osa[id_idx], urr[id_idx], thrs[id_idx]
        
        ax.plot(urr, osa, color=colors[idx], linestyle='-', label=labels[idx])
        if show_point[0]: # operational osa
            ax.scatter(op_urr, op_osa, marker='*', facecolors=colors[idx], edgecolors='black', zorder=20)
        if show_point[1]: # max osa
            ax.scatter(id_urr, id_osa, marker='d',facecolors=colors[idx], edgecolors='black', s=70, zorder=20)

        # Regional Zoom Plot
        if zoom[0][0]:
            axins.plot(urr, osa, color=colors[idx], linestyle='-', label=labels[idx])
            if show_point[0]: # operational osa
                axins.scatter(op_urr, op_osa, marker='*', facecolors=colors[idx], edgecolors='black', zorder=20)
            if show_point[1]: # max osa
                axins.scatter(id_urr, id_osa, marker='d',facecolors=colors[idx], edgecolors='black', s=70, zorder=20)
            ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=0.5)
            
    # Add custom legend item for markers
    handles, custom_labels = plt.gca().get_legend_handles_labels()
    if show_point[0]:
        handles.insert(0,legend_item_oosa)
        custom_labels.insert(0,'Optimal OSA')
    if show_point[1]:
        handles.insert(0,legend_item_iosa)
        custom_labels.insert(0,'max OSA')


    if lim != None:
        plt.xlim(lim[0])
        plt.ylim(lim[1])
    else:
        plt.xlim((-0.02,1.02))
    plt.title('Test Set: Known + Negative')
    plt.xlabel('URR')
    plt.ylabel('OSA')
    plt.grid(True)
    plt.legend(handles, custom_labels, loc='lower left')
    plt.show()

    # Plot OOSA for the test set with unknown samples
    fig, ax = plt.subplots(figsize=figsize)
    if zoom[1][0]:
        x1, x2, y1, y2 = zoom[1][1]  # subregion of the original image
        axins = ax.inset_axes(
            zoom[1][2] ,
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    else:
        axins = None
        
    for idx, d_i in enumerate(data_info):
        if show_val:
            ax.plot(urr, osa, color=colors[idx], alpha=0.1, linewidth=5)
        
        urr = eval_res[idx]['res'].test_unkn_urr
        osa = eval_res[idx]['res'].test_unkn_osa
        thrs = eval_res[idx]['res'].test_unkn_thrs

        op_idx = numpy.argmax(thrs > eval_res[idx]['op_thrs']) - 1
        op_osa, op_urr = osa[op_idx], urr[op_idx]
        id_idx = numpy.argmax(osa)
        id_osa, id_urr, id_thrs = osa[id_idx], urr[id_idx], thrs[id_idx]

        ax.plot(urr, osa, color=colors[idx], linestyle='-', label=labels[idx])
        if show_point[0]: # operational osa
            ax.scatter(op_urr, op_osa, marker='*', facecolors=colors[idx], edgecolors='black', s=70, zorder=20)
        if show_point[1]: # max osa
            ax.scatter(id_urr, id_osa,marker='d',facecolors=colors[idx], edgecolors='black', s=70, zorder=20)

        # Regional Zoom Plot
        if zoom[1][0]:
            axins.plot(urr, osa, color=colors[idx], linestyle='-', label=labels[idx])
            if show_point[0]: # operational osa
                axins.scatter(op_urr, op_osa, marker='*', facecolors=colors[idx], edgecolors='black', zorder=20)
            if show_point[1]: # max osa
                axins.scatter(id_urr, id_osa, marker='d',facecolors=colors[idx], edgecolors='black', s=70, zorder=20)
            ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=0.5)
            

    # Add custom legend item for markers
    handles, custom_labels = plt.gca().get_legend_handles_labels()
    if show_point[0]:
        handles.insert(0,legend_item_oosa)
        custom_labels.insert(0,'Optimal OSA')
    if show_point[1]:
        handles.insert(0,legend_item_iosa)
        custom_labels.insert(0,'max OSA')

    if lim != None:
        plt.xlim(lim[0])
        plt.ylim(lim[1])
    else:
        plt.xlim((-0.02,1.02))
    plt.title('Test Set: Known + Unknown')
    plt.xlabel('URR')
    plt.ylabel('OSA')
    plt.grid(True)
    plt.legend(handles, custom_labels, loc='lower left')
    plt.show()

def plot_OSCR(data_info, colors, figsize=(5,3), lim=None, show_val=True):
    
    eval_res = []
    for idx, d_i in enumerate(data_info):

        info = d_i['info']
        
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        res = eval_results(root_path)
        eval_res.append(res)
    
    
    plt.figure(figsize=figsize)
    for idx, d_i in enumerate(data_info):
        res = eval_res[idx]
        if show_val:
            plt.semilogx(res.val_fpr, res.val_ccr, color=colors[idx], alpha=0.1, linewidth=5)
        plt.semilogx(res.test_neg_fpr, res.test_neg_ccr, linestyle='-', color=colors[idx], label=d_i['label'])

    if lim != None:
        plt.xlim(lim[0])
        plt.ylim(lim[1])
    else:
        plt.xlim((0.8e-3,1.2))
    plt.title('Test Set: Negative')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Correct Classification Rate')
    plt.grid(True)
    plt.legend()


    plt.figure(figsize=figsize)
    for idx, d_i in enumerate(data_info):
        res = eval_res[idx]
        if show_val:
            plt.semilogx(res.val_fpr, res.val_ccr, color=colors[idx], alpha=0.1, linewidth=5)
        plt.semilogx(res.test_unkn_fpr, res.test_unkn_ccr, linestyle='-', color=colors[idx], label=d_i['label'])

    if lim != None:
        plt.xlim(lim[0])
        plt.ylim(lim[1])
    else:
        plt.xlim((0.8e-3,1.2))
    plt.title('Test Set: Unknown')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Correct Classification Rate')
    plt.grid(True)
    plt.legend()


def plot_confusion_mat(data_info, colors='viridis', figsize=(5,5), include_unknown=False, set_diag_mask=False, set_cmap_range=None, show_numbers=True, diag_sort=False):

    for d_i in data_info:

        info = d_i['info']
        
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        eval_res = eval_results(root_path)

        knowns = eval_res.test_unkn_gt != -1
        unknowns = ~knowns
        
        labels = numpy.unique(eval_res.test_unkn_gt[knowns])

        if include_unknown:
            threshold = 0.5
            gt = eval_res.test_unkn_gt
            pred = numpy.where(numpy.max(eval_res.test_unkn_probs, axis=1) >= threshold,
                               numpy.argmax(eval_res.test_unkn_probs, axis=1), -1)
            labels = numpy.append(labels, -1)
        else:
            known_gt = eval_res.test_unkn_gt[knowns]
            known_probs = eval_res.test_unkn_probs[knowns]
            known_pred = numpy.argmax(known_probs, axis=1)

            gt = known_gt
            pred = known_pred

        cm = confusion_matrix(gt, pred, labels=labels)
        if diag_sort:
            diag = numpy.diag(cm)
            idx=numpy.argsort(diag)[::-1]
            cm = cm[idx,:][:,idx]
        else:
            idx = range(cm.shape[0])
        plt.figure(figsize=figsize)
        if set_diag_mask: 
            mask = numpy.eye(cm.shape[0],dtype=int)
            cm = numpy.ma.masked_array(cm, mask=mask)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=colors, include_values=show_numbers, values_format='.0f')
        if not show_numbers:
            disp.ax_.set_xticklabels([])
            disp.ax_.set_yticklabels([])
        # else:
        #     disp.ax_.set_xticklabels(labels)
        #     disp.ax_.set_yticklabels(labels)
        if set_cmap_range != None:
            disp.im_.set_clim(vmin=set_cmap_range[0], vmax=set_cmap_range[1])

def plot_fpr_fnr_class(data_info, color=('red','green'), ylim=(0.0,0.5)):

    for d_i in data_info:
        info = d_i['info']
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        eval_res = eval_results(root_path)
        fpr_fnr_results, _, _, _, _ = compute_fpr_fnr(eval_res)
        # print(fpr_fnr_results)
        fig, axs = plt.subplots(2,1,figsize=(len(fpr_fnr_results)*0.3,2))
        for idx, c_fpr_fnr in enumerate(fpr_fnr_results):
            axs[0].bar(idx, c_fpr_fnr['fpr'], color=color[0], width=0.7)
            axs[1].bar(idx, c_fpr_fnr['fnr'], color=color[1], width=0.7)

        axs[0].set_ylabel('FPR')
        axs[0].set_xticks([])
        axs[0].set_ylim(ylim)

        axs[1].set_ylabel('FNR')
        axs[1].set_ylim(ylim)
        axs[1].set_xticks(range(len(fpr_fnr_results)), [])
        axs[1].set_xlabel('class')
        axs[1].set_yticklabels([label.get_text() if i > 0 else '' for i, label in enumerate(axs[0].get_yticklabels())])
        axs[1].invert_yaxis()
        plt.subplots_adjust(hspace=0)

def plot_fpr_fnr(data_info, hlines=[0.5, 3.5, 6.5], color=None, marker=None, figsize=(5,3), xlim=(-0.02, 0.52), has_oosa=None, is_verbose=True):

    if color == None:
        color = ['black'] * len(data_info)
    if marker == None:
        marker = ['o'] * len(data_info)

    fpr_fnr_results = []
    avg_results = []
    std_results = []
    for d_i in data_info:
        info = d_i['info']
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        eval_res = eval_results(root_path)
        res, avg, std, _, _ = compute_fpr_fnr(eval_res)
        fpr_fnr_results.append(res)
        avg_results.append(avg)
        std_results.append(std)

    print(f"average\tstd\tmodel")
    plt.figure(figsize=figsize)
    results = []
    for i, _ in enumerate(fpr_fnr_results):
        avg, std = avg_results[i], std_results[i]
        plt.errorbar(y=i, x=avg, xerr=std, color=color[i], capsize=3, fmt=marker[i])
        if i == 0:
            plt.fill_between([avg - std, avg + std], -1, len(data_info)+1, alpha=0.05, color=color[i])
            plt.vlines(avg,ymin=-1,ymax=len(data_info)+1,color=color[i],linestyles='dashed',alpha=0.2)
        results.append(avg)
        if is_verbose:
            print(f"{avg:.4f}\t{std:.3f}\t{data_info[i]['label']}")

    if xlim != None:
        plt.hlines(hlines, [xlim[0]]*len(hlines), [xlim[1]]*len(hlines), color='black', alpha=0.5)
        plt.xlim(xlim)

    plt.yticks(range(len(data_info)), [d_i['label'] for d_i in data_info])
    plt.ylim((-0.25,len(data_info)-0.75))
    plt.xlabel('|FPR - FNR|')
    # plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    if has_oosa:
        ax2 = plt.gca().twiny()
        ax2.plot(has_oosa['oosa'], range(len(data_info)), c=has_oosa['color'], marker='*', markersize=7)
        ax2.set_xlim(has_oosa['lim'])
        ax2.set_xlabel('OOSA')

    plt.tight_layout()

    return results

def _compute_fpr_fnr(eval_res):
        
    if eval_res.val_gt is None:
        res = [{'fpr': 0,'fnr':  0}]
    else:
        y_true = eval_res.test_unkn_gt
        y_probs = eval_res.test_unkn_probs
        y_pred = numpy.argmax(y_probs, axis=1)

        res = []
        classes = [uq for uq in numpy.unique(y_true) if uq >=0]
        for c in classes:
            # one-vs-rest
            y_true_c = y_true == c
            # y_pred_c = numpy.logical_and(y_pred == c, y_probs[range(len(y_pred)), y_pred] >= 0.5)
            y_pred_c = y_probs[range(len(y_pred)), c] >= 0.5
            # print(sum(y_pred_c))
            tn, fp, fn, tp = metrics.confusion_matrix(y_true_c, y_pred_c).ravel()
            res.append({'fpr': fp/(fp + tn),'fnr':  fn/(fn + tp)})

        fpr = numpy.array([r['fpr'] for r in res])
        fnr = numpy.array([r['fnr'] for r in res])
        fpr_avg, fpr_std = numpy.average(fpr), numpy.std(fpr)
        fnr_avg, fnr_std = numpy.average(fnr), numpy.std(fnr)

        diff = abs(fpr-fnr)
        diff_avg, diff_std, diff_min, diff_max = numpy.average(diff), numpy.std(diff), numpy.min(diff), numpy.max(diff)

    return {'fpr_avg':fpr_avg, 'fpr_std':fpr_std, 'fnr_avg':fnr_avg, 'fnr_std':fnr_std, 
            'diff_avg':diff_avg, 'diff_std':diff_std, 'diff_min':diff_min, 'diff_max':diff_max}

def compute_fpr_fnr(eval_res):
        
    if eval_res.val_gt is None:
        res = [{'fpr_nt': 0,'fpr_un':  0,'fnr':  0}]
    else:
        y_true = eval_res.test_unkn_gt
        y_probs = eval_res.test_unkn_probs
        y_pred = numpy.argmax(y_probs, axis=1)

        y_true_neg = eval_res.test_neg_gt
        y_probs_neg = eval_res.test_neg_probs
        y_pred_neg = numpy.argmax(y_probs_neg, axis=1)
        
        res = []
        classes = [uq for uq in numpy.unique(y_true) if uq >=0]
        for c in classes:
            # one-vs-rest
            y_true_c = y_true == c
            y_pred_c = y_probs[range(len(y_pred)), c] >= 0.5

            tmp_res = {}
            # FPR - Non-Target
            mask_1 = numpy.logical_or(y_true_c, y_true != -1)
            y_true_c_1 = y_true_c[mask_1]
            y_pred_c_1 = y_pred_c[mask_1]
            tn, fp, fn, tp = metrics.confusion_matrix(y_true_c_1, y_pred_c_1).ravel()
            tmp_res['fpr_nt'] = fp/(fp + tn)
            tmp_res['fnr'] = fn/(fn + tp)

            # FPR - Unknown
            mask_2 = numpy.logical_or(y_true_c, y_true == -1)
            y_true_c_2 = y_true_c[mask_2]
            y_pred_c_2 = y_pred_c[mask_2]
            tn, fp, fn, tp = metrics.confusion_matrix(y_true_c_2, y_pred_c_2).ravel()
            tmp_res['fpr_u'] = fp/(fp + tn)


            # FPR - Negative
            y_true_c = y_true_neg == c
            y_pred_c = y_probs_neg[range(len(y_pred_neg)), c] >= 0.5

            mask_2 = numpy.logical_or(y_true_c, y_true_neg == -1)
            y_true_c_2 = y_true_c[mask_2]
            y_pred_c_2 = y_pred_c[mask_2]
            tn, fp, fn, tp = metrics.confusion_matrix(y_true_c_2, y_pred_c_2).ravel()
            tmp_res['fpr_n'] = fp/(fp + tn)

            res.append(tmp_res)

        fpr_nt = numpy.array([r['fpr_nt'] for r in res])
        fpr_u = numpy.array([r['fpr_u'] for r in res])
        fpr_n = numpy.array([r['fpr_n'] for r in res])
        fnr = numpy.array([r['fnr'] for r in res])

        fpr_nt_avg, fpr_nt_std = numpy.average(fpr_nt), numpy.std(fpr_nt)
        fpr_u_avg, fpr_u_std = numpy.average(fpr_u), numpy.std(fpr_u)
        fpr_n_avg, fpr_n_std = numpy.average(fpr_n), numpy.std(fpr_n)
        fnr_avg, fnr_std = numpy.average(fnr), numpy.std(fnr)

    return {'fpr_nt_avg':fpr_nt_avg, 'fpr_nt_std':fpr_nt_std, 
            'fpr_u_avg':fpr_u_avg, 'fpr_u_std':fpr_u_std,
            'fpr_n_avg':fpr_n_avg, 'fpr_n_std':fpr_n_std,
            'fnr_avg':fnr_avg, 'fnr_std':fnr_std}

def plot_metrics(results:list, colors:list, color_labels:list, items:list, item_labels:list, item_ylims:list, xticks:list, figsize=(7,5)):

    assert len(items) <= 2, f"The number of items can be either 1 or 2. Given {len(items)} {items}"

    fig, ax1 = plt.subplots(figsize=figsize)
    # ax2 = ax1.twinx()
    handles = []
    for i in range(len(results)):
        result, color = results[i], colors[i]
        x = numpy.arange(len(xticks)) + 0.1 * i

        result_0, label_0, ylim_0 = result[items[0]], item_labels[0], item_ylims[0]
        p1 = ax1.scatter(x, result_0, color = color, marker='o', zorder=100)
        ax1.plot(x, result_0, color = color, ls='--', lw=1, alpha=0.5)
        h_p = (p1)
        if i == 0:
            ax1.set_xticks(ticks=x, labels=xticks)
            ax1.set_xlabel('# of negatives in training set')
            ax1.set_ylabel(label_0+' (o)')
            ax1.set_ylim(ylim_0)

        if len(items) == 2:
            result_1, label_1, ylim_1 = result[items[1]], item_labels[1], item_ylims[1]
            p2 = ax1.scatter(x, result_1, color = color, marker='x', zorder=100)
            ax1.plot(x, result_1, color = color, ls='--', lw=1, alpha=0.5)
            h_p = (p1,p2)

            ax1.set_ylabel(label_0 + ' (o)' + '  /  ' + label_1 + ' (x)')

            # result_1, label_1, ylim_1 = result[items[1]], item_labels[1], item_ylims[1]
            # p2 = ax2.scatter(x, result_1, color = color, marker='*', s=70, zorder=100)
            # ax2.plot(x, result_1, color = color, ls='--', lw=1, alpha=0.5)
            # h_p = (p1,p2)
            # if i == 0:
            #     ax2.set_ylabel(label_1+' (*)')
            #     ax2.set_ylim(ylim_1)

        handles.append(h_p)
        

    ax1.grid(axis='y')
    # fig.legend(handles, color_labels, loc='lower center', bbox_to_anchor=(0.5, 0.2), shadow=True, ncol=4, handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.legend(handles, color_labels, loc='upper center', bbox_to_anchor=(0.55, 0.2), shadow=True, ncol=4, handler_map={tuple: HandlerTuple(ndivide=None)})
    fig.tight_layout()

def print_metrics(data_info, show_osa_v=False, is_verbose=True):
    
    res = dict()

    # print("FPR↓\tFNR↓\tmaxOSA_N↑\tmaxOSA_U↑")
    if show_osa_v:
        print("FPR(NT)↓\tFPR(U)↓\tFPR(N)↓\tFNR↓\tmaxOSA_N↑\tmaxOSA_U↑\tmaxOSA_V↑")
    else:
        print("FPR(NT)↓\tFPR(U)↓\tFPR(N)↓\tFNR↓\tmaxOSA_N↑\tmaxOSA_U↑")

    for idx, d_i in enumerate(data_info):

        info = d_i['info']
        
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        eval_res = eval_results(root_path)
    
        if eval_res.val_gt is None:
            res_fpr_fnr = {'fpr_nt_avg':0, 'fpr_nt_std':0, 
                           'fpr_u_avg':0, 'fpr_u_std':0, 
                           'fpr_n_avg':0, 'fpr_n_std':0, 
                           'fnr_avg':0, 'fnr_std':0,}
            if show_osa_v:
                oosa = {'iosa_neg': 0, 'iosa_unkn':0, 'iosa_val':0}
            else:
                oosa = {'iosa_neg': 0, 'iosa_unkn':0}
        else: 
            # Class imbalance monitoring
            res_fpr_fnr = compute_fpr_fnr(eval_res) 

            # OSC Performance monitoring
            oosa = compute_oosa(eval_res.val_thrs, eval_res.val_osa, 
                                eval_res.test_neg_thrs, eval_res.test_neg_osa, 
                                eval_res.test_unkn_thrs, eval_res.test_unkn_osa)

        if is_verbose:
            if show_osa_v:
                print(f"{res_fpr_fnr['fpr_nt_avg']:.4f}\t{res_fpr_fnr['fpr_u_avg']:.4f}\t{res_fpr_fnr['fpr_n_avg']:.4f}\t{res_fpr_fnr['fnr_avg']:.4f}\t{oosa['iosa_neg']:.4f}\t{oosa['iosa_unkn']:.4f}\t{oosa['iosa_val']:.4f}")
            else:
                print(f"{res_fpr_fnr['fpr_nt_avg']:.4f}\t{res_fpr_fnr['fpr_u_avg']:.4f}\t{res_fpr_fnr['fpr_n_avg']:.4f}\t{res_fpr_fnr['fnr_avg']:.4f}\t{oosa['iosa_neg']:.4f}\t{oosa['iosa_unkn']:.4f}")

        if idx == 0:
            res['res_fpr_fnr'] = [res_fpr_fnr]
            res['oosa'] = [oosa]
        else:
            res['res_fpr_fnr'].append(res_fpr_fnr)
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

def compute_oosa(thrs_val, osa_val, thrs_neg, osa_neg, thrs_unkn, osa_unkn):
    op_thrs = thrs_val[numpy.argmax(osa_val)]
    iosa_val = numpy.max(osa_val)

    op_idx = numpy.argmax(thrs_neg > op_thrs) - 1
    oosa_neg = osa_neg[op_idx]
    iosa_neg = numpy.max(osa_neg)

    op_idx = numpy.argmax(thrs_unkn > op_thrs) - 1
    oosa_unkn = osa_unkn[op_idx]
    iosa_unkn = numpy.max(osa_unkn)

    return {'iosa_val': iosa_val, 'iosa_neg': iosa_neg, 'iosa_unkn': iosa_unkn, 'oosa_neg': oosa_neg, 'oosa_unkn':oosa_unkn}

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

def get_training_log(data_info, log_item = 'Loss/train'):

    info = data_info['info']

    log_path = f"./_models/{info[0]}/_s42/{info[1]}/{info[2]}/{info[3]}/Logs"
    onlyfiles = [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]
    log_file = onlyfiles[-1] # The last log

    event_acc = EventAccumulator(os.path.join(log_path, log_file))
    event_acc.Reload()

    logs = [e.value for e in event_acc.Scalars(log_item)]

    return logs



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

def _plot_score_dist(data_info, bins, colors, figsize=(10,3), ylim=None, plot_neg=True):

    center = (bins[:-1] + bins[1:]) / 2

    for idx in range(len(data_info)):
        plt.figure(figsize=figsize)

        info = data_info[idx]['info']

        # Load evaluation results
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        # folder_path = f"./_results/{item['info'][0]}/_s{s}/eval_{item['info'][1]}/{item['info'][2]}"
        results = eval_results(root_path)

        # Get Target and Non-target score distribution
        knowns = results.test_neg_gt != -1
        known_gt = results.test_neg_gt[knowns]
        known_score = results.test_neg_probs[knowns,:]

        target_mask = numpy.full(known_score.shape, False)
        target_mask[range(target_mask.shape[0]),known_gt] = True
        target_score = known_score[range(known_score.shape[0]), known_gt]
        
        non_target_score = numpy.reshape(known_score[~target_mask], (-1, known_score.shape[1]-1))
        non_target_max_score = numpy.max(non_target_score, axis=1)

        # Get Negatives and Unknown score distribution
        negatives = results.test_neg_gt == -1
        neg_score = results.test_neg_probs[negatives,:]
        neg_max_score = numpy.max(neg_score, axis=1)

        unknowns = results.test_unkn_gt == -1
        unkn_score = results.test_unkn_probs[unknowns,:]
        unkn_max_score = numpy.max(unkn_score, axis=1)

        # Get histogram data
        target_score_hist, _ = numpy.histogram(target_score, bins=bins, density=False)
        non_target_max_score_hist, _ = numpy.histogram(non_target_max_score, bins=bins, density=False)
        neg_max_score_hist, _ = numpy.histogram(neg_max_score, bins=bins, density=False)
        unkn_max_score_hist, _ = numpy.histogram(unkn_max_score, bins=bins, density=False)

        # Histogram data range from 0 to 1
        target_score_hist = 100 * target_score_hist/sum(target_score_hist)
        non_target_max_score_hist = 100 * non_target_max_score_hist/sum(non_target_max_score_hist)
        neg_max_score_hist = 100 * neg_max_score_hist/sum(neg_max_score_hist)
        unkn_max_score_hist = 100 * unkn_max_score_hist/sum(unkn_max_score_hist)
        # print(target_score_hist[:4], non_target_max_score_hist[:4])

        # plt.scatter(center, target_score_hist, color = colors[0], label='Target', marker='x')
        # plt.scatter(center, non_target_max_score_hist, color = colors[1], label='Non-target', marker='x')
        # plt.scatter(center, neg_max_score_hist, color = colors[2], label='Negative', marker='+')
        # plt.scatter(center, unkn_max_score_hist, color = colors[3],label='Unknown', marker='+')

        plt.plot(center, target_score_hist, color = colors[0], label='Target')
        plt.plot(center, non_target_max_score_hist, color = colors[1], label='Non-target')
        if plot_neg:
            plt.plot(center, neg_max_score_hist, color = colors[2], label='Negative')
        plt.plot(center, unkn_max_score_hist, color = colors[3],label='Unknown')


        if ylim == None:
            plt.ylim((0,100))
            plt.yscale('log')
        else:
            plt.ylim(ylim)
            plt.yscale('log')
        # plt.xticks([])
        # plt.ylabel('ratio')
        plt.xlabel('score')
        plt.title(data_info[idx]['label'])
        plt.grid(axis='both')
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
        plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())
        plt.tight_layout()


def plot_score_dist(data_info, bins, colors, figsize=(10,3), ylim=None, plot_neg=True):

    center = (bins[:-1] + bins[1:]) / 2

    for idx in range(len(data_info)):
        plt.figure(figsize=figsize)

        info = data_info[idx]['info']

        # Load evaluation results
        root_path = f'/home/user/hkim/UZH-MT/openset-binary/_results/{info[0]}/_s42/{info[1]}/eval_{info[2]}/{info[3]}'
        # folder_path = f"./_results/{item['info'][0]}/_s{s}/eval_{item['info'][1]}/{item['info'][2]}"
        results = eval_results(root_path)

        # Get Target and Non-target score distribution
        knowns = results.test_neg_gt != -1
        known_gt = results.test_neg_gt[knowns]
        known_score = results.test_neg_probs[knowns,:]

        target_mask = numpy.full(known_score.shape, False)
        target_mask[range(target_mask.shape[0]),known_gt] = True
        target_score = known_score[range(known_score.shape[0]), known_gt]
        
        non_target_score = numpy.reshape(known_score[~target_mask], (-1, known_score.shape[1]-1))
        non_target_max_score = numpy.max(non_target_score, axis=1)

        # Get Negatives and Unknown score distribution
        negatives = results.test_neg_gt == -1
        neg_score = results.test_neg_probs[negatives,:]
        neg_max_score = numpy.max(neg_score, axis=1)

        unknowns = results.test_unkn_gt == -1
        unkn_score = results.test_unkn_probs[unknowns,:]
        unkn_max_score = numpy.max(unkn_score, axis=1)

        # Get histogram data
        target_score_hist, _ = numpy.histogram(target_score, bins=bins, density=False)
        non_target_max_score_hist, _ = numpy.histogram(non_target_max_score, bins=bins, density=False)
        neg_max_score_hist, _ = numpy.histogram(neg_max_score, bins=bins, density=False)
        unkn_max_score_hist, _ = numpy.histogram(unkn_max_score, bins=bins, density=False)

        # Histogram data range from 0 to 1
        target_score_hist = 100 * target_score_hist/sum(target_score_hist)
        non_target_max_score_hist = 100 * non_target_max_score_hist/sum(non_target_max_score_hist)
        neg_max_score_hist = 100 * neg_max_score_hist/sum(neg_max_score_hist)
        unkn_max_score_hist = 100 * unkn_max_score_hist/sum(unkn_max_score_hist)
        print(target_score_hist[:4], non_target_max_score_hist[:4])

        if plot_neg:
            plt.scatter(center, neg_max_score_hist, color = colors[2], label='Negative', marker='_', linewidths=2)
        plt.scatter(center, unkn_max_score_hist, color = colors[3],label='Unknown', marker='_', linewidths=2)
        plt.scatter(center, non_target_max_score_hist, color = colors[1], label='Non-target', marker='_', linewidths=2)
        plt.scatter(center, target_score_hist, color = colors[0], label='Target', marker='+', linewidths=2)

        # plt.plot(center, target_score_hist, color = colors[0], label='Target')
        # plt.plot(center, non_target_max_score_hist, color = colors[1], label='Non-target')
        # if plot_neg:
        #     plt.plot(center, neg_max_score_hist, color = colors[2], label='Negative')
        # plt.plot(center, unkn_max_score_hist, color = colors[3],label='Unknown')


        if ylim == None:
        #     plt.ylim((0,100))
            plt.yscale('log')
        else:
            plt.ylim(ylim)
            plt.yscale('log')
        # plt.xticks([])
        # plt.ylabel('ratio')
        plt.xlabel('score')
        plt.title(data_info[idx]['label'])
        plt.grid(axis='both')
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
        plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())
        plt.tight_layout()


def __plot_dist_prob(data_info, num_classes, bins, seeds, figsize=(10,3), ylim=(5e-7, 1.5)):

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

    net.eval()
    with torch.no_grad():
        for (x, y) in tqdm(data_loader, miniters=int(len(data_loader)/3), maxinterval=600, disable=not is_verbose):
            
            logs, feat = net(device(x))

            gt.extend(y.tolist())
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

def get_openset_perf(test_gt:numpy.array, test_probs:numpy.array, unkn_gt_label=-1, is_verbose=False):

    # vary thresholds
    ccr, fpr = [], []
    kn_probs = test_probs[test_gt != unkn_gt_label]
    unkn_probs = test_probs[test_gt == unkn_gt_label]
    gt = test_gt[test_gt != unkn_gt_label]

    # Get CCR and FPR
    thresholds = sorted(numpy.append(kn_probs[range(len(gt)),gt], numpy.max(unkn_probs, axis=1)))
    for tau in tqdm(thresholds, miniters=int(len(gt)/5), maxinterval=600, disable=not is_verbose):
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

    return (ccr, fpr, urr, osa, thresholds)

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
                if i == 2 and not save_feats:
                    continue
                numpy.save(os.path.join(root, f'{key}_{pred_name[i]}.npy'), arr)
    
    print(f"Prediction Saved Successfully!\n{root}\n")

def save_openset_perf(category:str, ccr:list, threshold:list, fpr:list, urr:list, osa:list, root:str):

    if not os.path.exists(root):
        os.makedirs(root)
        print(f"Folder '{root}' created successfully.")

    numpy.save(os.path.join(root, f'{category}_ccr.npy'), ccr)
    numpy.save(os.path.join(root, f'{category}_thrs.npy'), threshold)
    numpy.save(os.path.join(root, f'{category}_fpr.npy'), fpr)
    numpy.save(os.path.join(root, f'{category}_urr.npy'), urr)
    numpy.save(os.path.join(root, f'{category}_osa.npy'), osa)

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
