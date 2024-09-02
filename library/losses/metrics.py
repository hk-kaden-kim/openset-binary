import torch
from torch.nn import functional as F
from .. import tools

"""This file contains different metrics that can be applied to evaluate the training"""


def multi_binary_confidence(logits:torch.Tensor, target):
    """Measures the multi binary confidence of the correct class for known samples,
        ...
    """

    with torch.no_grad():
        num_of_classes = logits.shape[1]
        enc_target = tools.target_encoding(target, num_of_classes, init=1, kn_target=0)
        all_probs = F.sigmoid(logits)

        confidence_by_sample = torch.mean(torch.abs(enc_target - all_probs), dim=1)
        confidence = torch.sum(confidence_by_sample)

    return torch.tensor((confidence, len(logits), confidence, len(logits)))

# #######################################################################
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
# #######################################################################


def accuracy(prediction, target):
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
                torch.max(prediction[known], axis=1).indices == target[known], dtype=int
            )
        else:
            correct = 0

    return torch.tensor((correct, total))


def sphere(representation, target, sphere_radius=None):
    """Computes the radius of unknown samples.
    For known samples, the radius is computed and added only when sphere_radius is not None.

    Parameters:

      representation: the feature vector of the samples

      target: the vector of true classes; can be -1 for unknown samples

    Returns a tensor with two entries:

      length: The sum of the length of the samples

      total: The total number of considered samples
    """

    with torch.no_grad():
        known = target >= 0

        magnitude = torch.norm(representation, p=2, dim=1)

        sum = torch.sum(magnitude[~known])
        total = torch.sum(~known)

        if sphere_radius is not None:
            sum += torch.sum(torch.clamp(sphere_radius - magnitude, min=0.0))
            total += torch.sum(known)

    return torch.tensor((sum, total))


def confidence(logits, target, negative_offset=0.1):
    """Measures the softmax confidence of the correct class for known samples,
    and 1 + negative_offset - max(confidence) for unknown samples.

    Parameters:

      logits: the output of the network, must be logits

      target: the vector of true classes; can be -1 for unknown samples

    Returns a tensor with two entries:

      confidence: the sum of the confidence values for the samples

      total: The total number of considered samples
    """

    with torch.no_grad():
        known = target >= 0

        pred = torch.nn.functional.softmax(logits, dim=1)
        #    import ipdb; ipdb.set_trace()

        confidence = 0.0
        if torch.sum(known):
            confidence += torch.sum(pred[known, target[known]])
        if torch.sum(~known):
            confidence += torch.sum(
                1.0 + negative_offset - torch.max(pred[~known], dim=1)[0]
            )

    return torch.tensor((confidence, len(logits)))

def confidence_v2(logits, target, offset=0., unknown_class = -1, last_valid_class = None, is_binary=False):
# def confidence(scores, target_labels, offset=0., unknown_class = -1, last_valid_class = None):
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
        if is_binary:
            scores = F.sigmoid(logits)
        else:
            scores = torch.nn.functional.softmax(logits, dim=1)
        unknown = target == unknown_class
        known = torch.logical_and(target >= 0, ~unknown)
        kn_count = sum(known).item()    # Total known samples in data
        neg_count = sum(unknown).item()  # Total negative samples in data
        kn_conf_sum = 0.0
        neg_conf_sum = 0.0
        if kn_count:
            # Average confidence known samples
            kn_conf_sum = torch.sum(scores[known, target[known]]).item()
        if neg_count:
            # we have negative labels in the validation set
            neg_conf_sum = torch.sum(
                1.0
                + offset
                - torch.max(scores[unknown,:last_valid_class], dim=1)[0]
            ).item()

    return torch.tensor((kn_conf_sum, kn_count, neg_conf_sum, neg_count))
