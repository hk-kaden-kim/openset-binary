import torch
import torchvision
from torch.nn import functional as F
from .. import tools

def calc_MOON_weights(num_of_classes, labels, init_val=1, is_verbose=False):

    # initialize
    counts = len(labels) 
    pos_weights = torch.empty(num_of_classes).fill_(init_val)
    neg_weights = torch.empty(num_of_classes).fill_(init_val)
    
    # get the target distribution
    if counts%2:
        pos_tar_dist = (counts+1)//2
    else:
        pos_tar_dist = counts//2
    neg_tar_dist = counts - pos_tar_dist  

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
        if pos_tar_dist > dist:
            pos_weights[u_label[i]] = 1
        else:
            pos_weights[u_label[i]] = neg_src_dist[i] / dist

    # set negative weights
    for i, dist in enumerate(neg_src_dist):
        if neg_tar_dist > dist:
            neg_weights[u_label[i]] = 1
        else:
            neg_weights[u_label[i]] = pos_src_dist[i] / dist

    if is_verbose:
        print(f"Positive Weights:\n{pos_weights}")
        print(f"Negative Weights:\n{neg_weights}")
        print()

    return tools.device(pos_weights), tools.device(neg_weights)

class multi_binary_loss:

    def __init__(self, num_of_classes=10, gt_labels=None, loss_config=None,):

        self.num_of_classes = num_of_classes
        self.loss_config = loss_config

        if self.loss_config.option == 'moon':
            print(f"Using Multi-Binary Classifier Loss with MOON Paper Version.")
            self.weight_global = loss_config.moon_weight_global
            self.weigith_init_val = loss_config.moon_weight_init_val
            self.unkn_weight = loss_config.moon_unkn_weight
            if self.weight_global:
                print(f"Loss weights are calculated globally.")
                self.glob_pos_weights, self.glob_neg_weights = calc_MOON_weights(self.num_of_classes, gt_labels, self.weigith_init_val, is_verbose=True)
                # self.glob_pos_weights, self.glob_neg_weights = torch.ones((num_of_classes,)), torch.ones((num_of_classes,)) 
            else:
                print(f"Loss weights will be calculated in every batch.")
                self.glob_pos_weights, self.glob_neg_weights = None, None
        
        elif self.loss_config.option == 'focal':
            print(f"Using Multi-Binary Classifier Loss with Focal Loss Version.")
            self.alpha = loss_config.focal_alpha
            self.gamma = loss_config.focal_gamma
        
        else:
            print(f"Using the nomal version of Multi-Binary Classifier Loss.")

    @tools.loss_reducer
    def __call__(self, logit_values, target_labels):
        
        # Encode target values
        all_target = tools.target_encoding(target_labels, self.num_of_classes)

        # Calculate probabilities for each sample
        all_probs = F.sigmoid(logit_values)


        if self.loss_config.option == 'moon':
            # Approach 1. MOON : Weighting by source and target element distribution.
            # Get positive or negative weights for each classifer
            if self.weight_global:
                pos_weights, neg_weights = self.glob_pos_weights, self.glob_neg_weights
            else:
                pos_weights, neg_weights = calc_MOON_weights(self.num_of_classes, target_labels, self.weigith_init_val, is_verbose=False)
            
            # Create weight matrix for each sample
            weights = torch.where(all_target==1, pos_weights, neg_weights)

            # Multiply additional unknown sample weight
            for idx, t in enumerate(target_labels):
                if t == -1: # Check unknown samples
                    weights[idx] = weights[idx] * self.unkn_weight

            # assign newly created tensor to gpu if cuda is available
            if torch.cuda.is_available():
                gpu = all_probs.get_device()
                weights = weights.to(gpu)

            all_loss = F.binary_cross_entropy(all_probs, all_target, weights)
        
        elif self.loss_config.option == 'focal':
            # Approach 2. Focal Loss : More weight on high confidence FN and less weight on low confidence TN.
            all_loss = torchvision.ops.sigmoid_focal_loss(logit_values, all_target,
                                                          alpha=self.loss_config.focal_alpha,
                                                          gamma=self.loss_config.focal_gamma,
                                                          reduction='mean')

        else:
            # Base : Simple use of binary cross entropy loss.
            all_loss = F.binary_cross_entropy(all_probs, all_target)

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
