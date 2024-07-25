import os
import torch
import numpy
from scipy.interpolate import interp1d
from tqdm import tqdm

from ..architectures import architectures
from ..tools import device, set_device_cpu, get_device, print_table

# set_device_cpu()

########################################################################
# Reference Code
# 
# Author: Manuel Günther
# Date: 2024
# Availability: https://gitlab.uzh.ch/manuel.guenther/eos-example
########################################################################
    
def load_network(args, config, which, num_classes):

    network_file = os.path.join(config.arch.model_root, f"{args.scale}/{args.arch}/{which}")
    
    # if config.arch.force_fc_dim == 2 and args.scale == 'SmallScale':
    #     network_file = os.path.join(config.arch.model_root, f"{args.scale}_fc_dim_2/{args.arch}/{which}")

    if config.data.largescale.level > 1 and args.scale == 'LargeScale':
        network_file = os.path.join(config.arch.model_root, f"{args.scale}_{config.data.largescale.level}/{args.arch}/{which}")

    if config.need_sync:
        network_file = os.path.join(network_file, f"{which}.pth")
    else:
        network_file = os.path.join(network_file, f"{which}.model")

    if os.path.exists(network_file):

        # Add bias term at the last layer, if it is either 'Garbage' and 'MultiBinary'
        final_layer_bias = False 
        if which in ['MultiBinary']:    # Garbage
        # if True:    # Garbage
            final_layer_bias = True
        # assert False, f"{final_layer_bias} {args.approach}"
        if 'LeNet_plus_plus' in args.arch:
            arch_name = 'LeNet_plus_plus'
        elif 'ResNet_18' in args.arch:
            arch_name = 'ResNet_18'
        elif 'ResNet_50' in args.arch:
            arch_name = 'ResNet_50'
        else:
            arch_name = None

        net = architectures.__dict__[arch_name](use_BG=which=="Garbage",
                                                num_classes=num_classes,
                                                final_layer_bias=final_layer_bias,)
        checkpoint = torch.load(network_file, map_location=torch.device('cpu')) 

        if config.need_sync:
            print('Weights are came from the reference code! Sync the weight name!')
            checkpoint = architectures.checkpoint_sync(checkpoint["model_state_dict"], map_location=torch.device('cpu'))     
        
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
        for (x, y) in tqdm(data_loader, miniters=int(len(data_loader)/5), maxinterval=600, disable=not is_verbose):
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

def get_oscr_curve(test_gt:numpy.array, test_probs:numpy.array, unkn_gt_label=-1, at_fpr=0.01, is_verbose=False):

    # vary thresholds
    ccr, fpr = [], []
    kn_probs = test_probs[test_gt != unkn_gt_label]
    unkn_probs = test_probs[test_gt == unkn_gt_label]
    gt = test_gt[test_gt != unkn_gt_label]

    for tau in tqdm(sorted(kn_probs[range(len(gt)),gt]), miniters=int(len(gt)/5), maxinterval=600, disable=not is_verbose):
        # correct classification rate
        ccr.append(numpy.sum(numpy.logical_and(
            numpy.argmax(kn_probs, axis=1) == gt,
            kn_probs[range(len(gt)),gt] >= tau
        )) / len(kn_probs))
        # false positive rate for validation and test set
        fpr.append(numpy.sum(numpy.max(unkn_probs, axis=1) >= tau) / len(unkn_probs))

    if at_fpr is not None:
        print(f"\tCCR@FPR{at_fpr} : {find_ccr_at_fpr(numpy.array(fpr),numpy.array(ccr),at_fpr)}")

    return (ccr, fpr)

def eval_pred_save(pred_results:dict, root:str, save_feats=True):

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

def oscr_save(ccr:list, fpr_neg:list, fpr_unkn:list, root:str):

    if not os.path.exists(root):
        os.makedirs(root)
        print(f"Folder '{root}' created successfully.")

    numpy.save(os.path.join(root, 'ccr.npy'), ccr)
    numpy.save(os.path.join(root, 'fpr_neg.npy'), fpr_neg)
    numpy.save(os.path.join(root, 'fpr_unkn.npy'), fpr_unkn)

    print(f"OSCR Data Saved Successfully!\n{root}\n")

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
