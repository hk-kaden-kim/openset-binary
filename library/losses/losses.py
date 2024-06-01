import torch
from torch.nn import functional as F
from .. import tools

class multi_binary_loss:

    def __init__(self, num_of_classes=10):
        self.num_of_classes = num_of_classes
        # self.eye = tools.device(torch.eye(self.num_of_classes))
        # self.ones = tools.device(torch.ones(self.num_of_classes))
        # self.unknowns_multiplier = 1.0 / self.num_of_classes

    @tools.loss_reducer
    def __call__(self, logit_values, target):
        
        # Encode target values
        all_target = tools.target_encoding(target, self.num_of_classes)

        # Calculate probabilities for each sample
        all_probs = F.sigmoid(logit_values)

        # Get balancing weights
        t_weights = torch.ones(self.num_of_classes) # TODO: Task-Balance
        s_weights = torch.ones(logit_values.shape[0]) # TODO: Class-Balance
    
        # Calculate Loss by each binary classifier
        for j in range(all_probs.shape[1]):
            probs = all_probs[:,j]
            targets = all_target[:,j]
            task_loss = (t_weights[j] * F.binary_cross_entropy(probs, targets, s_weights)).reshape(-1)
            if j == 0:
                all_loss = task_loss
            else:
                all_loss = torch.concat([all_loss, task_loss])

        return all_loss


########################################################################
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

class entropic_openset_loss:

    def __init__(self, num_of_classes=10):
        self.num_of_classes = num_of_classes
        self.eye = tools.device(torch.eye(self.num_of_classes))
        self.ones = tools.device(torch.ones(self.num_of_classes))
        self.unknowns_multiplier = 1.0 / self.num_of_classes

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
        return sample_loss

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
        return loss
