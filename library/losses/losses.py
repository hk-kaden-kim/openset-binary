import torch
import torchvision
from torch.nn import functional as F
import numpy as np
from .schedule import g_convex, g_concave, g_linear, g_composite_1, g_composite_2
from .. import tools

PRV_EPOCH = 0
def get_PRV_EPOCH():
    return PRV_EPOCH
def set_PRV_EPOCH(epoch):
    global PRV_EPOCH
    PRV_EPOCH = epoch


def calc_class_cnt(num_of_classes, labels, is_verbose=False):

    all_counts = len(labels) 
    pos_cnts = torch.Tensor([0]*num_of_classes)

    # get each known label counts = positives
    labels = labels[labels != -1]
    u_labels, cnts = labels.unique(return_counts=True)
    u_labels = u_labels.to(torch.int)

    for idx, l in enumerate(u_labels):
        pos_cnts[l] = cnts[idx]
    pos_cnts = pos_cnts.to(torch.int)
    
    if is_verbose:
        tools.print_table(u_labels.cpu().numpy(), pos_cnts.cpu().numpy())

    # get other label counts : negatives
    neg_cnts = all_counts - pos_cnts

    return tools.device(pos_cnts), tools.device(neg_cnts)

def calc_class_ratio(num_of_classes, labels, init_val=1, is_verbose=False):

    # initialize
    pos_ratio = torch.Tensor([init_val]*num_of_classes)
    neg_ratio = torch.Tensor([init_val]*num_of_classes)

    # get the source distribution
    pos_cnts, neg_cnts = calc_class_cnt(num_of_classes, labels, is_verbose)

    # u_label, pos_src_cnt = labels.unique(return_counts=True)
    # u_label, pos_src_cnt = u_label.to(torch.int), pos_src_cnt.to(torch.int)
    # if is_verbose:
    #     tools.print_table(u_label.cpu().numpy(), pos_src_cnt.cpu().numpy())

    # if -1 in u_label:
    #     u_label, pos_src_cnt = u_label[1:], pos_src_cnt[1:]
    # neg_src_cnt = counts - pos_src_cnt

    # get positive and negative ratios
    # if there is no positive samples for the class in the batch, 0 weighted for all samples in this class.
    for i, p_cnt in enumerate(pos_cnts):
        n_cnt = neg_cnts[i]
        if n_cnt > p_cnt:
            pos_ratio[i] = 1
            neg_ratio[i] = p_cnt/n_cnt
        else:
            pos_ratio[i] = n_cnt/p_cnt
            neg_ratio[i] = 1

    if is_verbose:
        print(f"Positive Ratio:\n{pos_ratio}")
        print(f"Negative Ratio:\n{neg_ratio}")
        print()

    return tools.device(pos_ratio), tools.device(neg_ratio)

def get_class_weights(pos_ratio, neg_ratio, labels, enc_labels, unkn_weight, epoch, epochs, schedule='None'):

    # is_verbose = False
    # if epoch != get_PRV_EPOCH():
    #     set_PRV_EPOCH(epoch)
    #     is_verbose = True

    # if is_verbose:
    #     print()
    #     print(pos_ratio, neg_ratio)

    if schedule == 'linear':
        pos_ratio = torch.stack([g_linear(epoch, 1, w, 1, epochs) for w in pos_ratio]).view(-1)
        neg_ratio = torch.stack([g_linear(epoch, 1, w, 1, epochs) for w in neg_ratio]).view(-1)
    elif schedule == 'convex':
        pos_ratio = torch.stack([g_convex(epoch, 1, w, 1, epochs) for w in pos_ratio]).view(-1)
        neg_ratio = torch.stack([g_convex(epoch, 1, w, 1, epochs) for w in neg_ratio]).view(-1)
    elif schedule == 'concave':
        pos_ratio = torch.stack([g_concave(epoch, 1, w, 1, epochs) for w in pos_ratio]).view(-1)
        neg_ratio = torch.stack([g_concave(epoch, 1, w, 1, epochs) for w in neg_ratio]).view(-1)
    elif schedule == 'composite_1':
        pos_ratio = torch.stack([g_composite_1(epoch, 1, w, 1, epochs) for w in pos_ratio]).view(-1)
        neg_ratio = torch.stack([g_composite_1(epoch, 1, w, 1, epochs) for w in neg_ratio]).view(-1)
    elif schedule == 'composite_2':
        pos_ratio = torch.stack([g_composite_2(epoch, 1, w, 1, epochs) for w in pos_ratio]).view(-1)
        neg_ratio = torch.stack([g_composite_2(epoch, 1, w, 1, epochs) for w in neg_ratio]).view(-1)

    # if is_verbose:
    #     print(pos_ratio, neg_ratio)

    # Create weight matrix for each sample
    weights = torch.where(enc_labels==1, pos_ratio, neg_ratio)

    # Multiply additional unknown sample weight
    for idx, t in enumerate(labels):
        if t == -1 and unkn_weight != 1: # Check unknown samples
            weights[idx] = weights[idx] * unkn_weight
    
    return tools.device(weights)

def get_mining_mask(logits, enc_labels, pos_cnts, neg_cnts, alpha, epoch, epochs, schedule='None'):

    # is_verbose = False
    # if epoch != get_PRV_EPOCH():
    #     set_PRV_EPOCH(epoch)
    #     is_verbose = True

    # Leave probabilities only for negative samples
    neg_probs = torch.where(enc_labels!=1, F.sigmoid(logits), -1)

    # Create masking of the mining result
    k_mask = torch.zeros(enc_labels.shape)
    k_mask = tools.device(k_mask)

    for i, p_cnt in enumerate(pos_cnts):
        n_cnt = neg_cnts[i]

        # Mining all negatives if there is no positives
        if p_cnt == 0:
            k_mask[:,i] == 1
            continue
        
        # Get k for the mining
        if schedule == 'linear':
            c_k = g_linear(epoch, n_cnt, p_cnt, 1, epochs)
        elif schedule == 'convex':
            c_k = g_convex(epoch, n_cnt, p_cnt, 1, epochs)
        elif schedule == 'concave':
            c_k = g_concave(epoch, n_cnt, p_cnt, 1, epochs)
        elif schedule == 'composite_1':
            c_k = g_composite_1(epoch, n_cnt, p_cnt, 1, epochs)
        elif schedule == 'composite_2':
            c_k = g_composite_2(epoch, n_cnt, p_cnt, 1, epochs)
        else:
            c_k = alpha*(n_cnt - p_cnt) + p_cnt      # NEW ...

        # Mining all negatives if k is larger than # of negatives 
        if c_k > n_cnt:
            k_mask[:,i] = 1
            continue
        
        # Hard Negative Minings : Negatives with a high probability
        c_k_idxs = torch.topk(neg_probs[:,i], int(c_k)).indices
        k_mask[c_k_idxs, i] = 1

        # if is_verbose:
        #     set_PRV_EPOCH(epoch)
        #     print(c_k, p_cnt, n_cnt, alpha, epoch, epochs)

    # Get the final masks including all postives and mined negatives
    mining_mask = torch.where(torch.logical_or(enc_labels==1, k_mask==1), 1, 0)

    return tools.device(mining_mask)

def focal_loss_custom(logits, labels, enc_labels, gamma, alpha_pos, alpha_neg, unkn_weight, epoch, epochs, schedule='None', focal_mining=1, reduction='mean'):
    # Reference > torchvision.ops.sigmoid_focal_loss

    # Get p_t
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, enc_labels, reduction="none")
    p_t = p * enc_labels + (1 - p) * (1 - enc_labels)

    # Gamma scheduling
    # if schedule == 'linear':
    #     gamma = g_linear(epoch, 0, gamma, 1, epochs)
    # elif schedule == 'convex':
    #     gamma = g_concave(epoch, 0, gamma, 1, epochs)
    # elif schedule == 'concave':
    #     gamma = g_convex(epoch, 0, gamma, 1, epochs)
    # elif schedule == 'composite_1':
    #     gamma = g_composite_2(epoch, 0, gamma, 1, epochs)
    # elif schedule == 'composite_2':
    #     gamma = g_composite_1(epoch, 0, gamma, 1, epochs)

    # Get focal loss without class weighting
    loss = ce_loss * ((1 - p_t) ** gamma)

    # if epoch != get_PRV_EPOCH():
    #     set_PRV_EPOCH(epoch)

    # Variant 1. Get focal loss with class weighting
    if (torch.concat((alpha_pos, alpha_neg)) != 1).sum() > 0:
        alpha_t = get_class_weights(alpha_pos, alpha_neg, labels, enc_labels, unkn_weight, epoch, epochs, schedule)
        loss = alpha_t * loss

    # Variant 2. Get focal loss with Hard Negative Mining
    if focal_mining != 1:
        pos_cnts, neg_cnts = calc_class_cnt(enc_labels.shape[1], labels)
        mining_mask = get_mining_mask(logits, enc_labels, pos_cnts, neg_cnts, focal_mining, epoch, epochs, schedule)
        loss = mining_mask * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

    return loss

def focal_loss_custom_2(logits, enc_labels, gamma, alpha_t, mining_mask, reduction='mean'):
    # Reference > torchvision.ops.sigmoid_focal_loss

    # Get p_t
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, enc_labels, reduction="none")
    p_t = p * enc_labels + (1 - p) * (1 - enc_labels)

    # Get focal loss without class weighting
    loss = ce_loss * ((1 - p_t) ** gamma)


    # Variant 1. Get focal loss with class weighting
    if (alpha_t != 1).sum() > 0:
        # print(f"Establish Class Weighting! {(alpha_t != 1).sum()} {(mining_mask != 1).sum()}")
        loss = alpha_t * loss
    # else:
    #     print(f"No Class Weighting!")

    # Variant 2. Get focal loss with Hard Negative Mining
    if (mining_mask != 1).sum() > 0:
        # print(f"Hard Negative Mining! {(alpha_t != 1).sum()} {(mining_mask != 1).sum()}")
        loss = mining_mask * loss
    # else:
    #     print(f"No Hard Negative Mining!")

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

    return loss


class multi_binary_loss:

    def __init__(self, epochs, num_of_classes=10, gt_labels=None, loss_config=None,is_verbose=True):

        # Common
        self.num_of_classes = num_of_classes
        self.loss_config = loss_config
        self.schedule = loss_config.schedule
        self.epochs = epochs
        self.glob_pos_ratio, self.glob_neg_ratio = None, None

        # Class-weighting Hyperparameter
        self.wcls_type = loss_config.wclass_type
        self.wcls_w_unkn = loss_config.wclass_weight_unkn
        self.wcls_w_init = 1

        # Mining Hyperparameter
        # self.dist_global = loss_config.mining_dist_global     # Not Used
        self.mining_a = loss_config.mining_alpha

        # Focal Hyperparameter
        self.focal_g = loss_config.focal_gamma
        self.focal_a = loss_config.focal_alpha
        self.focal_uw = loss_config.focal_weight_unkn
        self.focal_mining_a = loss_config.focal_mining
        
        # Console output
        if self.schedule != 'None':
            print(f"Scheduling : {self.schedule}")
        
        if self.loss_config.option == 'wclass':
            if is_verbose:
                print(f"BCE Loss & CLASS WEIGHTING !!!")
                print(f"Unknown Weights : {self.wcls_w_unkn}")
                print(f"Class Weights : {self.wcls_type}")
            if self.wcls_type == 'global':
                self.glob_pos_ratio, self.glob_neg_ratio = calc_class_ratio(num_of_classes, gt_labels, 1, is_verbose=is_verbose)

        elif self.loss_config.option == 'mining':
            if is_verbose:
                print(f"BCE Loss & HARD NEG MINING !!!")
                print(f"Alpha : {self.mining_a}")

        elif self.loss_config.option == 'focal':
            if is_verbose:
                print(f"BCE Loss -> FOCAL LOSS !!!")
                print(f"Gamma : {self.focal_g}")
                print(f"Alpha : {self.focal_a}")
                print(f"Unknown Weights: {self.focal_uw}")
            if self.focal_a == 'global':
                self.glob_pos_ratio, self.glob_neg_ratio = calc_class_ratio(num_of_classes, gt_labels, 1, is_verbose=is_verbose)
            if is_verbose:
                print(f"Mining Alpha : {self.focal_mining_a}")

        else:
            if is_verbose:
                print(f"BCE (Baseline)")

        if is_verbose:
            print()
        
    @tools.loss_reducer
    def __call__(self, logit_values, target_labels, epoch):
        
        # Encode target values
        enc_target_labels = tools.target_encoding(target_labels, self.num_of_classes)

        # ------------------------------------------------------
        # Approach 1. BCE Loss & CLASS WEIGHTING
        # ------------------------------------------------------
        if self.loss_config.option == 'wclass':

            # Get positive or negative weights for each classifer
            pos_ratio, neg_ratio = self.glob_pos_ratio, self.glob_neg_ratio
            if (pos_ratio == None) and (neg_ratio == None):
                pos_ratio, neg_ratio = calc_class_ratio(self.num_of_classes, target_labels, self.wcls_w_init, is_verbose=False)

            weights = get_class_weights(pos_ratio, neg_ratio, target_labels, enc_target_labels, self.wcls_w_unkn, epoch, self.epochs, self.schedule)

            all_loss = F.binary_cross_entropy_with_logits(logit_values, enc_target_labels, weights)

        # ------------------------------------------------------
        # Approach 2. BCE Loss & HARD NEG MINING
        # ------------------------------------------------------
        elif self.loss_config.option == 'mining':

            pos_cnts, neg_cnts = calc_class_cnt(self.num_of_classes, target_labels, is_verbose=False)

            weights = get_mining_mask(logit_values, enc_target_labels, pos_cnts, neg_cnts, self.mining_a, epoch, self.epochs, self.schedule)

            all_loss = F.binary_cross_entropy_with_logits(logit_values, enc_target_labels, weights)

        # ------------------------------------------------------
        # Approach 3. BCE Loss -> FOCAL LOSS
        # ------------------------------------------------------
        elif self.loss_config.option == 'focal':

            # alpha_pos, alpha_neg = self.glob_pos_ratio, self.glob_neg_ratio
            # if (alpha_pos == None) and (alpha_neg == None):
            #     alpha_pos, alpha_neg = calc_class_ratio(self.num_of_classes, target_labels, self.wcls_w_init, is_verbose=False)
                
            # all_loss = focal_loss_custom(logit_values, target_labels, enc_target_labels, self.focal_g,
            #                              alpha_pos, alpha_neg, self.focal_uw, 
            #                              epoch, self.epochs, self.schedule, 
            #                              focal_mining = self.focal_mining_a)

            # Variant 1. Get focal loss with class weighting
            if self.focal_a == 'None':
                alpha_t = tools.device(torch.ones(logit_values.shape))
            else:
                alpha_pos, alpha_neg = self.glob_pos_ratio, self.glob_neg_ratio                
                if self.focal_a == 'batch': # Alpha pos and neg need to be calculated in a batch-wise.
                    alpha_pos, alpha_neg = calc_class_ratio(self.num_of_classes, target_labels, self.wcls_w_init, is_verbose=False)           
                alpha_t = get_class_weights(alpha_pos, alpha_neg, target_labels, enc_target_labels, self.focal_uw, epoch, self.epochs, self.schedule)

            # Variant 2. Get focal loss with Hard Negative Mining
            if self.focal_mining_a == 1:       # No mining
                mining_mask = tools.device(torch.ones(logit_values.shape))
            else:
                pos_cnts, neg_cnts = calc_class_cnt(self.num_of_classes, target_labels)
                mining_mask = get_mining_mask(logit_values, enc_target_labels, pos_cnts, neg_cnts, self.focal_mining_a, epoch, self.epochs, self.schedule)


            # Calculate Focal Loss
            all_loss = focal_loss_custom_2(logit_values, enc_target_labels, self.focal_g, alpha_t, mining_mask)
            
        # ------------------------------------------------------
        # (BASE) Approach.
        # ------------------------------------------------------
        else:
            # Base : Simple use of binary cross entropy loss.
            all_loss = F.binary_cross_entropy_with_logits(logit_values, enc_target_labels)

        return all_loss

########################################################################
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

class entropic_openset_loss:

    def __init__(self, num_of_classes=10, unkn_weight=1):
        self.num_of_classes = num_of_classes
        self.eye = tools.device(torch.eye(self.num_of_classes))
        self.ones = tools.device(torch.ones(self.num_of_classes))
        self.unknowns_multiplier = unkn_weight / self.num_of_classes

    @tools.loss_reducer
    def __call__(self, logit_values, target, sample_weights=None):
        catagorical_targets = tools.device(torch.zeros(logit_values.shape))
        known_indexes = target != -1
        unknown_indexes = ~known_indexes
        catagorical_targets[known_indexes, :] = self.eye[target[known_indexes]]
        catagorical_targets[unknown_indexes, :] = (
            self.ones.expand((torch.sum(unknown_indexes).item(), self.num_of_classes))
            * self.unknowns_multiplier
        )
        log_values = F.log_softmax(logit_values, dim=1)
        negative_log_values = -1 * log_values
        loss = negative_log_values * catagorical_targets
        sample_loss = torch.sum(loss, dim=1)
        if sample_weights is not None:
            sample_loss = sample_loss * sample_weights
        return sample_loss.mean()

class objectoSphere_loss:
    def __init__(self, knownsMinimumMag=50.0):
        self.knownsMinimumMag = knownsMinimumMag

    @tools.loss_reducer
    def __call__(self, features, target, sample_weights=None):
        # compute feature magnitude
        mag = features.norm(p=2, dim=1)
        # For knowns we want a certain magnitude
        mag_diff_from_ring = torch.clamp(self.knownsMinimumMag - mag, min=0.0)

        # Loss per sample
        loss = tools.device(torch.zeros(features.shape[0]))
        known_indexes = target != -1
        unknown_indexes = ~known_indexes
        # knowns: punish if magnitude is inside of ring
        loss[known_indexes] = mag_diff_from_ring[known_indexes]
        # unknowns: punish any magnitude
        loss[unknown_indexes] = mag[unknown_indexes]
        loss = torch.pow(loss, 2)
        if sample_weights is not None:
            loss = sample_weights * loss
        return loss.mean()
