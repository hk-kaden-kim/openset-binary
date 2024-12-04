import os
import torch
import numpy
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D

import sklearn.metrics as metrics

from ..architectures import architectures
from ..tools import device, print_table

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import warnings
warnings.filterwarnings("ignore")


###############################
# Generate evaluation results
###############################

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

def load_network(args, config, which, num_classes, seed=-1):

    network_file = os.path.join(config.arch.model_root, f"{args.scale}/_s{seed}/{args.arch}/{which}")
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
        net = architectures.__dict__[arch_name](num_classes=num_classes,
                                                final_layer_bias=False,
                                                feat_dim=config.arch.feat_dim)
        
        checkpoint = torch.load(network_file, map_location=torch.device('cpu')) 

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


###############################
# Post-processing
###############################

def print_metrics(data_info, show_osa_v=False, is_verbose=True):
    
    res = dict()

    # print("FPR↓\tFNR↓\tmaxOSA_N↑\tmaxOSA_U↑")
    if show_osa_v:
        print("maxOSA_V↑\tmaxOSA_N↑\tmaxOSA_U↑\tFNR(1e-1)↓\tFPR(NT)↓\tFPR(N)↓\tFPR(U)↓")
    else:
        print("maxOSA_N↑\tmaxOSA_U↑\tFNR(1e-1)↓\tFPR(NT)↓\tFPR(N)↓\tFPR(U)↓")

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
                print(f"{oosa['iosa_val']:.4f}\t{oosa['iosa_neg']:.4f}\t{oosa['iosa_unkn']:.4f}\t{10*res_fpr_fnr['fnr_avg']:.4f}\t{10*res_fpr_fnr['fpr_nt_avg']:.4f}\t{10*res_fpr_fnr['fpr_n_avg']:.4f}\t{10*res_fpr_fnr['fpr_u_avg']:.4f}")
            else:
                print(f"{oosa['iosa_neg']:.4f}\t{oosa['iosa_unkn']:.4f}\t{10*res_fpr_fnr['fnr_avg']:.4f}\t{10*res_fpr_fnr['fpr_nt_avg']:.4f}\t{10*res_fpr_fnr['fpr_n_avg']:.4f}\t{10*res_fpr_fnr['fpr_u_avg']:.4f}")

        if idx == 0:
            res['res_fpr_fnr'] = [res_fpr_fnr]
            res['oosa'] = [oosa]
        else:
            res['res_fpr_fnr'].append(res_fpr_fnr)
            res['oosa'].append(oosa)

    return res

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
    plt.title('OSA Plot\nwith Testset Negative ($D_K \cup D_N$)')
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
    plt.title('OSA Plot\nwith Testset Unknown ($D_K \cup D_U$)')
    plt.xlabel('URR')
    plt.ylabel('OSA')
    plt.grid(True)
    plt.legend(handles, custom_labels, loc='lower left')
    plt.show()

def plot_score_dist(data_info, bins, colors, title, figsize=(10,3), ylim=None, plot_neg=True):

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
            plt.bar(center, neg_max_score_hist, color = colors[2], label='Negative', width=bins[1]-bins[0],edgecolor='black')
        plt.bar(center, unkn_max_score_hist, color = colors[3],label='Unknown', width=bins[1]-bins[0],edgecolor='black')
        plt.bar(center, non_target_max_score_hist, color = colors[1], label='Non-target', width=bins[1]-bins[0],edgecolor='black')
        plt.bar(center, target_score_hist, color = colors[0], label='Target', width=bins[1]-bins[0],edgecolor='black', hatch='//')

        plt.ylim(ylim)
        plt.xlabel('score')
        if title[idx] == None:
            plt.title(data_info[idx]['label'])
        else:
            plt.title(title[idx])
        plt.grid(axis='y')
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))
        plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())
        plt.tight_layout()

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

def get_training_log(data_info, log_item = 'Loss/train'):

    info = data_info['info']

    log_path = f"./_models/{info[0]}/_s42/{info[1]}/{info[2]}/{info[3]}/Logs"
    onlyfiles = [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]
    log_file = onlyfiles[-1] # The last log

    event_acc = EventAccumulator(os.path.join(log_path, log_file))
    event_acc.Reload()

    logs = [e.value for e in event_acc.Scalars(log_item)]

    return logs

