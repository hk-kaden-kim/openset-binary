import torch
from torch.nn import functional as F

"""This file contains different metrics that can be applied to evaluate the training"""

########################################################################
# Reference: 
# Vision And Security Technology (VAST) Lab in UCCS
# https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

def accuracy(scores, target):
    """Computes the classification accuracy of the classifier based on known samples only.
    Any target that does not belong to a certain class (target is -1) is disregarded.

    Parameters:

      prediction: the output of the network, can be logits or softmax scores

      target: the vector of true classes; can be -1 for unknown samples

    Returns a tensor with two entries:

      correct: The number of correctly classified samples

      total: The total number of considered samples
    """

    with torch.no_grad():
        known = target >= 0

        total = torch.sum(known, dtype=int)
        if total:
            correct = torch.sum(
                torch.max(scores[known], axis=1).indices == target[known], dtype=int
            )
        else:
            correct = 0

    return torch.tensor((correct, total))

def confidence(scores:torch.Tensor, labels:torch.Tensor, offset=0., unknown_class = -1, last_valid_class = None):
    """ Returns model's confidence, Taken from https://github.com/Vastlab/vast/tree/main/vast.

    Args:
        scores(tensor): Softmax scores of the samples.
        target_labels(tensor): Target label of the samples.
        offset(float): Confidence offset value, typically 1/number_of_classes.
        unknown_class(int) which index to consider as unknown
        last_valid_class(int or None) which classes to predict; can be None for all and -1 for BG approach

    Returns:
        kn_conf: Confidence of known samples.
        kn_count: Count of known samples.
        neg_conf: Confidence of negative samples.
        neg_count Count of negative samples.
    """
    with torch.no_grad():

        unknown = labels == unknown_class
        known = torch.logical_and(labels >= 0, ~unknown)

        kn_count = sum(known).item()    # Total known samples in data
        neg_count = sum(unknown).item()  # Total negative samples in data
        kn_conf_sum = 0.0
        neg_conf_sum = 0.0

        if kn_count:
            # Sum confidence known samples
            kn_conf_sum = torch.sum(scores[known, labels[known]]).item()
        if neg_count:
            # we have negative labels in the validation set
            neg_conf_sum = torch.sum(
                1.0
                + offset
                - torch.max(scores[unknown,:last_valid_class], dim=1)[0]
            ).item()

    return torch.tensor((kn_conf_sum, kn_count, neg_conf_sum, neg_count))