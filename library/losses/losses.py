import torch
import torchvision
from torch.nn import functional as F
import numpy as np
from .schedule import g_poly_convex, g_poly_concav, g_linear, g_composite
from .. import tools

def calc_mining_dist(target:torch.Tensor):

    c_dist = []
    for c in range(target.shape[1]):
        if target[:,c].sum() == 0: # No positives
            c_dist.append([torch.tensor(0)] * 3)
            continue
        try:
            _, (neg_cnt, pos_cnt) = torch.unique(target[:,c], return_counts=True)
        except:
            assert False, f"{torch.unique(target[:,c], return_counts=True)} {c}"
        ratio = neg_cnt/pos_cnt     # S- / S+
        c_dist.append([ratio, neg_cnt, pos_cnt])
    
    return torch.Tensor(c_dist)

def get_weights_k_hard_neg_mining_3(probs:torch.Tensor, target:torch.Tensor, c_dist:torch.Tensor, alpha, epoch, epochs, schedule='None'):

    # If it is scheduled
    if schedule == 'linear':
        alpha = 1 - g_linear(epoch, epochs)
    elif schedule == 'convex':
        alpha = 1 - g_poly_concav(epoch, epochs)
    elif schedule == 'concave':
        alpha = 1 - g_poly_convex(epoch, epochs)
    elif schedule == 'composite':
        alpha = 1 - g_composite(epoch, epochs)

    # Replace Prob. of Positives to -1 for easy process
    neg_probs = torch.where(target!=1, probs, -1) 

    # Create masking of the mining result
    k_mask = torch.zeros(neg_probs.shape, device=tools.get_device())
    for c in range(target.shape[1]):
        ratio, neg_cnt, pos_cnt = c_dist[c,:]

        if pos_cnt == 0: # No positives
            k_mask[:,c] = 1
            continue

        c_k = int((ratio ** (alpha)) * pos_cnt)     # Get alpha for classifier c
        if c_k < neg_cnt:   # Mining Hard Negatives only If # of mining sample is smaller than # of negatives
            c_k_indices = torch.topk(neg_probs[:,c], c_k).indices
            k_mask[c_k_indices, c] = 1
        else:
            k_mask[:,c] = 1

    weights = torch.where(torch.logical_or(target==1, k_mask==1), 1, 0)
    return weights

# def get_weights_k_hard_neg_mining_2(probs:torch.Tensor, target:torch.Tensor, alpha, epoch, epochs, schedule='None'):

#     # If it is scheduled
#     if schedule == 'linear':
#         alpha = 1 - g_linear(epoch, epochs)
#     elif schedule == 'convex':
#         alpha = 1 - g_poly_concav(epoch, epochs)
#     elif schedule == 'concave':
#         alpha = 1 - g_poly_convex(epoch, epochs)
#     elif schedule == 'composite':
#         alpha = 1 - g_composite(epoch, epochs)

#     # Replace Prob. of Positives to -1 for easy process
#     neg_probs = torch.where(target!=1, probs, -1) 

#     # Get k_mask
#     k_mask = torch.zeros(neg_probs.shape, device=tools.get_device())
#     for c in range(target.shape[1]):
#         if target[:,c].sum() == 0: # No positives
#             k_mask[:,c] = 1
#             continue
#         try:
#             _, (neg_cnt, pos_cnt) = torch.unique(target[:,c], return_counts=True)
#         except:
#             assert False, f"{torch.unique(target[:,c], return_counts=True)} {c}"
#         ratio = neg_cnt/pos_cnt     # S- / S+

#         c_k = int((ratio ** (alpha)) * pos_cnt)     # Get alpha for classifier c
#         if c_k < neg_cnt:   # Mining Hard Negatives only If alpha is greater than the number of negatives
#             c_k_indices = torch.topk(neg_probs[:,c], c_k).indices
#             k_mask[c_k_indices, c] = 1
#         else:
#             k_mask[:,c] = 1

#     weights = torch.where(torch.logical_or(target==1, k_mask==1), 1, 0)

#     return weights

# def get_weights_k_hard_neg_mining(probs, target, alpha):

#     # Replace Prob. of Positives to -1 for easy process
#     neg_probs = torch.where(target!=1, probs, -1) 

#     # Pick alpha hard negative samples' indices by each class classifier
#     if neg_probs.shape[0] < alpha:
#         alpha = neg_probs.shape[0] # Just in case the last batch has less than alpha samples.
#     try:
#         k_mining_indices = torch.topk(neg_probs, alpha, dim=0).indices
#     except:
#         print(neg_probs.shape, alpha)
#         assert False, "Error!"

#     # Create a mask for hard negative mining result
#     k_mining_mask = torch.zeros(target.shape, device=tools.get_device())
#     for c, idx in enumerate(k_mining_indices.T):
#         k_mining_mask[idx, c] = 1

#     weights = torch.where(torch.logical_or(target==1, k_mining_mask==1), 1, 0)

#     return weights

def get_weights_moon(pos_weights, neg_weights, target, enc_target, unkn_weight, epoch, epochs, schedule='None'):

    # If it is scheduled
    if schedule == 'linear':
        pos_weights = pos_weights ** g_linear(epoch, epochs)
        neg_weights = neg_weights ** g_linear(epoch, epochs)
    elif schedule == 'convex':
        pos_weights = pos_weights ** g_poly_convex(epoch, epochs)
        neg_weights = neg_weights ** g_poly_convex(epoch, epochs)
    elif schedule == 'concave':
        pos_weights = pos_weights ** g_poly_concav(epoch, epochs)
        neg_weights = neg_weights ** g_poly_concav(epoch, epochs)
    elif schedule == 'composite':
        pos_weights = pos_weights ** g_composite(epoch, epochs)
        neg_weights = neg_weights ** g_composite(epoch, epochs)

    # Create weight matrix for each sample
    weights = torch.where(enc_target==1, pos_weights, neg_weights)

    # Multiply additional unknown sample weight
    for idx, t in enumerate(target):
        if t == -1: # Check unknown samples
            weights[idx] = weights[idx] * unkn_weight
    
    return tools.device(weights)

def calc_moon_weights(num_of_classes, labels, init_val=1, is_verbose=False):

    # initialize
    counts = len(labels) 
    pos_weights = torch.empty(num_of_classes).fill_(init_val)
    neg_weights = torch.empty(num_of_classes).fill_(init_val)

    # get the source distribution
    u_label, pos_src_dist = labels.unique(return_counts=True)
    u_label, pos_src_dist = u_label.to(torch.int), pos_src_dist.to(torch.int)
    if is_verbose:
        tools.print_table(u_label.cpu().numpy(), pos_src_dist.cpu().numpy())

    if -1 in u_label:
        u_label, pos_src_dist = u_label[1:], pos_src_dist[1:]
    neg_src_dist = counts - pos_src_dist

    # set positive weights
    for i, dist in enumerate(pos_src_dist):
        if neg_src_dist[i] > dist:
            pos_weights[u_label[i]] = 1
        else:
            pos_weights[u_label[i]] = (neg_src_dist[i] / dist)

    # set negative weights
    for i, dist in enumerate(neg_src_dist):
        if pos_src_dist[i] > dist:
            neg_weights[u_label[i]] = 1
        else:
            neg_weights[u_label[i]] = (pos_src_dist[i] / dist)

    if is_verbose:
        print(f"Positive Weights:\n{pos_weights}")
        print(f"Negative Weights:\n{neg_weights}")
        print()

    return tools.device(pos_weights), tools.device(neg_weights)


class multi_binary_loss:

    def __init__(self, epochs, num_of_classes=10, gt_labels=None, loss_config=None,):

        self.num_of_classes = num_of_classes
        self.loss_config = loss_config
        self.schedule = loss_config.schedule
        self.epochs = epochs
        self.weight_global = loss_config.moon_weight_global
        self.weigith_init_val = loss_config.moon_weight_init_val
        self.unkn_weight = loss_config.moon_unkn_weight
        self.alpha = loss_config.focal_alpha
        self.gamma = loss_config.focal_gamma
        self.dist_global = loss_config.mining_dist_global
        self.alpha = loss_config.mining_alpha

        if self.loss_config.option == 'moon':
            print(f"Using Multi-Binary Classifier Loss with MOON Paper Version.")
            if self.weight_global:
                print(f"Loss weights are calculated globally.")
                self.glob_pos_weights, self.glob_neg_weights = calc_moon_weights(self.num_of_classes, gt_labels, self.weigith_init_val, is_verbose=True)
            else:
                print(f"Loss weights will be calculated in every batch.")
                self.glob_pos_weights, self.glob_neg_weights = None, None

            if self.schedule != 'None':
                print(f"Weights are applied with {self.schedule} scheduling")

            if self.unkn_weight != 1:
                print(f"Unknown loss weighting = {self.unkn_weight}")
        
        elif self.loss_config.option == 'focal':
            print(f"Using Multi-Binary Classifier Loss with Focal Loss Version.")

        elif self.loss_config.option == 'mining':
            print(f"Using Hard Negative Mining")
            if self.dist_global:
                print(f"# of hard negatives is set globally.")
                all_target_enc = tools.target_encoding(gt_labels, self.num_of_classes)
                self.glob_c_dist = calc_mining_dist(all_target_enc)
                print(f"ratio\tnegatvies\tpositives\n{self.glob_c_dist}")
            else:
                print(f"# of hard negatives is set by each batch.")
                self.glob_num_mining = None

            if self.schedule == 'None':
                print(f"alpha = {self.alpha}")
            else:
                print(f"alpha scheduling : {self.schedule} ")
        else:
            print(f"Using the nomal version of Multi-Binary Classifier Loss.")

    @tools.loss_reducer
    def __call__(self, logit_values, target_labels, epoch):
        
        # Encode target values
        all_target_enc = tools.target_encoding(target_labels, self.num_of_classes)

        # Calculate probabilities for each sample. Shape: [batch_size, # of classes]
        all_probs = F.sigmoid(logit_values)

        # ------------------------------------------------------
        # Approach 1. MOON : Weighting by source and target element distribution.
        # ------------------------------------------------------
        if self.loss_config.option == 'moon':

            # Get positive or negative weights for each classifer
            if self.weight_global:
                pos_weights, neg_weights = self.glob_pos_weights, self.glob_neg_weights
            else:
                pos_weights, neg_weights = calc_moon_weights(self.num_of_classes, target_labels, self.weigith_init_val, is_verbose=False)
            
            weights = get_weights_moon(pos_weights, neg_weights, target_labels, all_target_enc, self.unkn_weight, epoch, self.epochs, self.schedule)
            all_loss = F.binary_cross_entropy(all_probs, all_target_enc, weights)

        # ------------------------------------------------------
        # Approach 2. Focal Loss : More weight on high confidence FN and less weight on low confidence TN.
        # ------------------------------------------------------
        elif self.loss_config.option == 'focal':

            # Calculate Focal Loss
            all_loss = torchvision.ops.sigmoid_focal_loss(logit_values, all_target_enc,
                                                          alpha=self.alpha,
                                                          gamma=self.gamma,
                                                          reduction='mean')
        
        # ------------------------------------------------------
        # Approach 3. Hard Negative Mining : ...
        # ------------------------------------------------------
        elif self.loss_config.option == 'mining':

            # Create a weight matrix to conisder samples either all positives or alpha hard negatives.
            if self.dist_global:
                c_dist = calc_mining_dist(all_target_enc)
                c_dist[:,0] = self.glob_c_dist[:,0] # Keep only global level neg/pos ratio
            else:
                c_dist = calc_mining_dist(all_target_enc)
            # weights = get_weights_k_hard_neg_mining(all_probs, all_target_enc, self.topk)
            # weights = get_weights_k_hard_neg_mining_2(all_probs, all_target_enc, self.alpha, epoch, self.epochs, self.schedule)
            weights = get_weights_k_hard_neg_mining_3(all_probs, all_target_enc, c_dist, self.alpha, epoch, self.epochs, self.schedule)
            all_loss = F.binary_cross_entropy(all_probs, all_target_enc, weights) 

        # ------------------------------------------------------
        # (BASE) Approach.
        # ------------------------------------------------------
        else:
            # Base : Simple use of binary cross entropy loss.
            all_loss = F.binary_cross_entropy(all_probs, all_target_enc)

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
