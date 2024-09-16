import os
import numpy as np
import itertools
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
from .. import losses

colors_global = np.array(
    [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
        '#000000',
    ]
)
# colors_global = colors_global / 255.0

def get_probs(pnts, which, net, config, gpu=None):

    pnts = torch.tensor(pnts).float()
    if gpu is not None and torch.cuda.is_available():
        pnts = pnts.to(f'{gpu}')

    result = net.deep_feature_forward(pnts)
    if which == 'OvR':
        probs = F.sigmoid(result).detach()
    elif which == 'OpenSetOvR':
        osovr_act = losses.OpenSetOvR(config.osovr_sigma)
        probs = osovr_act(result, net.fc2.weight.data).detach()
    else:
        probs = F.softmax(result, dim=1).detach()
    probs = torch.max(probs, dim=1).values

    if gpu is not None and torch.cuda.is_available():
        probs = probs.cpu()
    
    return probs

def deep_features_plot(which, net, config, gpu, unkn_gt_label, pred_results, results_dir):

    print('Plot Deep Feature Space!')
    train_gt, _, train_feats, _ = pred_results['train']
    test_neg_gt, _, test_neg_feats, _ = pred_results['test_neg']
    test_unkn_gt, _, test_unkn_feats, _ = pred_results['test_unkn']

    # Deep Feature Plotting : Training Samples
    print("Training Set...")
    known_tag = train_gt != unkn_gt_label
    unknown_tag = ~known_tag

    pos_features = train_feats[known_tag]
    labels = train_gt[known_tag]
    neg_features = train_feats[unknown_tag]
    
    plotter_2D(pos_features, labels, 
                         neg_features=None, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/1_{}_train.{}')

    plotter_2D(pos_features, labels, 
                         neg_features=None, heat_map=True, 
                         final=True, file_name=str(results_dir)+'/2_{}_heat_train.{}',
                         which=which, net=net, gpu=gpu, config=config)

    plotter_2D(pos_features, labels, 
                         neg_features=neg_features, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/3_{}_train_neg.{}')
    print("Done!")

    # Deep Feature Plotting : Testing Samples (+ Negatives)
    print("Test Set with 'Known Unknown Samples'...")
    known_tag = test_neg_gt != unkn_gt_label
    unknown_tag = ~known_tag

    pos_features = test_neg_feats[known_tag]
    labels = test_neg_gt[known_tag]
    neg_features = test_neg_feats[unknown_tag]

    plotter_2D(pos_features, labels, 
                         neg_features=None, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/1_{}_test.{}')

    plotter_2D(pos_features, labels,
                         neg_features=None, heat_map=True, 
                         final=True, file_name=str(results_dir)+'/2_{}_heat_test.{}',
                         which=which, net=net, gpu=gpu, config=config)

    plotter_2D(pos_features, labels, 
                         neg_features=neg_features, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/3_{}_test_neg.{}'),
    print("Done!")

    # Deep Feature Plotting : Testing Samples (+ Unknowns)
    print("Test Set with 'Unknown Unknown Samples'...")
    known_tag = test_unkn_gt != unkn_gt_label
    unknown_tag = ~known_tag

    pos_features = test_unkn_feats[known_tag]
    labels = test_unkn_gt[known_tag]
    neg_features = test_unkn_feats[unknown_tag]

    plotter_2D(pos_features, labels, 
                         neg_features=neg_features, heat_map=False, 
                         final=True, file_name=str(results_dir)+'/3_{}_test_unkn.{}')

    print("Done!\n")
########################################################################
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

def plot_histogram(
    pos_features,
    neg_features,
    pos_labels="Knowns",
    neg_labels="Unknowns",
    title="Histogram",
    file_name="{}foo.pdf",
):
    """
    This function plots the Histogram for Magnitudes of feature vectors.
    """
    pos_mag = np.sqrt(np.sum(np.square(pos_features), axis=1))
    neg_mag = np.sqrt(np.sum(np.square(neg_features), axis=1))

    pos_hist = np.histogram(pos_mag, bins=500)
    neg_hist = np.histogram(neg_mag, bins=500)

    fig, ax = plt.subplots(figsize=(4.5, 1.75))
    ax.plot(
        pos_hist[1][1:],
        pos_hist[0].astype(np.float16) / max(pos_hist[0]),
        label=pos_labels,
        color="g",
    )
    ax.plot(
        neg_hist[1][1:],
        neg_hist[0].astype(np.float16) / max(neg_hist[0]),
        color="r",
        label=neg_labels,
    )

    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.xscale("log")
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    if file_name:
        plt.savefig(file_name.format("Hist", "pdf"), bbox_inches="tight")
    plt.show()
    plt.close()
    
def plotter_2D(
    pos_features,
    labels,
    neg_features=None,
    pos_labels="Knowns",
    neg_labels="Unknowns",
    title=None,
    file_name="foo.pdf",
    final=False,
    heat_map=False,
    prob_function=get_probs,
    *args,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    if heat_map:
        try:
            # min_x, max_x = np.min(pos_features[:, 0]), np.max(pos_features[:, 0])
            # min_y, max_y = np.min(pos_features[:, 1]), np.max(pos_features[:, 1])
            max_x = np.max(np.abs(pos_features[:, 0]))
            max_y = np.max(np.abs(pos_features[:, 1]))
            xy_range = max(max_x, max_y).item()
            x = np.linspace(-1 * xy_range * 1.5, xy_range * 1.5, 500)
            y = np.linspace(-1 * xy_range * 1.5, xy_range * 1.5, 500)
            pnts = list(itertools.chain(itertools.product(x, y)))
            pnts = np.array(pnts)

            res = prob_function(pnts, *args, **kwargs)

            heat_map = ax.pcolormesh(
                x,
                y,
                np.array(res).reshape(500, 500).transpose(),
                # cmap='gray',
                rasterized=True,
                shading="auto",
                vmin=0.0,
                vmax=1.0,
            )
            fig.colorbar(heat_map, ax=ax, fraction=0.046, pad=0.04)
        except Exception as error:
            print("An exception occurred:", error)
    
    colors = colors_global
    if neg_features is not None:
        # Remove black color from knowns
        colors = colors_global[:-1]

    # The following code segment needs to be improved
    colors_with_repetition = colors.tolist()
    for i in range(int(len(set(labels.tolist())) / colors.shape[0])):
        colors_with_repetition.extend(colors.tolist())
    colors_with_repetition.extend(
        colors.tolist()[: int(colors.shape[0] % len(set(labels.tolist())))]
    )
    colors_with_repetition = np.array(colors_with_repetition)

    labels_to_int = np.zeros(labels.shape[0])
    for i, l in enumerate(set(labels.tolist())):
        labels_to_int[labels == l] = i

    ax.scatter(
        pos_features[:, 0],
        pos_features[:, 1],
        c=colors_with_repetition[labels_to_int.astype(int)],
        edgecolors="none",
        s=5,
    )
    if neg_features is not None:
        ax.scatter(
            neg_features[:, 0],
            neg_features[:, 1],
            c="k", alpha=0.2,
            edgecolors="none",
            s=15,
            marker="*",
        )
    if final:
        fig.gca().spines["right"].set_position("zero")
        fig.gca().spines["bottom"].set_position("zero")
        fig.gca().spines["left"].set_visible(False)
        fig.gca().spines["top"].set_visible(False)
        ax.tick_params(
            axis="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        ax.axis("equal")
    try:
        fig.savefig(file_name.format("2D_plot", "png"), bbox_inches="tight")
    except Exception as error:
        print("An exception occurred:", error)
        # fig.savefig(file_name.format("2D_plot", "pdf"), bbox_inches="tight")
    fig.show()
    if neg_features is not None:
        try:
            plot_histogram(
                pos_features,
                neg_features,
                pos_labels=pos_labels,
                neg_labels=neg_labels,
                title=title,
                file_name=file_name.format("hist", "pdf"),
            )
        except Exception as error:
            print("An exception occurred:", error)
            
    plt.close()

def plot_OSRC(to_plot, no_of_false_positives=None, filename=None, title=None):
    """
    :param to_plot: list of tuples containing (knowns_accuracy,OSE,label_name)
    :param no_of_false_positives: To write on the x axis
    :param filename: filename to write
    :return: None
    """
    fig, ax = plt.subplots()
    if title is not None:
        fig.suptitle(title, fontsize=20)
    for plot_no, (knowns_accuracy, OSE, label_name) in enumerate(to_plot):
        ax.plot(OSE, knowns_accuracy, label=label_name)
    ax.set_xscale("log")
    ax.autoscale(enable=True, axis="x", tight=True)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Correct Classification Rate", fontsize=18, labelpad=10)
    if no_of_false_positives is not None:
        ax.set_xlabel(
            f"False Positive Rate : Total Unknowns {no_of_false_positives}",
            fontsize=18,
            labelpad=10,
        )
    else:
        ax.set_xlabel("False Positive Rate", fontsize=18, labelpad=10)
    ax.legend(
        loc="lower center", bbox_to_anchor=(-1.25, 0.0), ncol=1, fontsize=18, frameon=False
    )
    # ax.legend(loc="upper left")
    if filename is not None:
        if "." not in filename:
            filename = f"{filename}.pdf"
        fig.savefig(f"{filename}", bbox_inches="tight")
    plt.show()

    plt.close()
