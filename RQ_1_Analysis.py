
import os
import re
import itertools
from tqdm import tqdm

import numpy as np

import torch
from torch.nn import functional as F

import sklearn.metrics as metrics
import sklearn.preprocessing as prep

import matplotlib.pyplot as plt
import matplotlib as mpl

from library import architectures, tools, evals, dataset, losses

labels={
  "SoftMax" : "Plain SoftMax",
  "Garbage" : "Garbage Class",
  "EOS" : "Entropic Open-Set",
  "OvR" : "One-vs-Rest Classifiers",
  "OpenSetOvR": "Open-Set OvR Classifiers"
}

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='...TBD'
    )

    parser.add_argument("--scale", "-sc", required=True, choices=['SmallScale', 'LargeScale_1', 'LargeScale_2', 'LargeScale_3'], help="Choose the scale of evaluation dataset.")
    parser.add_argument("--arch", "-ar", required=True)
    parser.add_argument("--approach", "-ap", nargs="+", default=list(labels.keys()), choices=list(labels.keys()), help = "Select the approaches to evaluate; non-existing models will automatically be skipped")
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()


def list_model_files(folder_path):
    # List to store .model files
    model_files = []

    # Iterate over all files in the given folder
    for file_name in os.listdir(folder_path):
        # Check if the file ends with .model
        if file_name.endswith('.model'):
            model_files.append(file_name)

    return model_files



def deep_feat_viz_sample_progress(appr, seed, arch, scale, is_verbose=False, save_plot=None):

    assert "plus_plus" in arch, f"Check the given approach : {appr}. It could have more than 2 features."

    # load dataset
    if scale == 'SmallScale':
        data = dataset.EMNIST('/local/scratch/hkim',
                              split_ratio = 0.8, seed = seed,
                              convert_to_rgb=False)
        batch_size = 2048
    else:
        assert False, "NOT SUPPORTED!"

    # Load evaluation dataset
    if appr == 'Garbage':
        train_set_neg, _, num_classes = data.get_train_set(is_verbose=True, size_train_negatives=0, has_background_class=True)
        unkn_gt_label = num_classes
    else:
        train_set_neg, _, num_classes = data.get_train_set(is_verbose=True, size_train_negatives=0, has_background_class=False)
        unkn_gt_label = -1
    data_loader = torch.utils.data.DataLoader(train_set_neg, batch_size=batch_size, shuffle=False)

    # Find all check points
    network_folder = f'../../_models/SmallScale/_s{seed}/{arch}/{appr}'

    models = list_model_files(network_folder)
    models = [m for m in models if re.compile(r'_\d').search(m)]
    models.sort(key=lambda f: int(re.sub('\D', '', f)))

    num_r = int(np.ceil(len(models)/5))
    num_c = 5
    fig, ax = plt.subplots(num_r,num_c,figsize=(num_c*4,num_r*4), sharex=False, sharey=False)

    for idx, m in enumerate(models):
        i,j = idx // num_c, idx% num_c

        network_file = os.path.join(network_folder, m)

        # Collecting heatmap data
        net = architectures.__dict__['LeNet_plus_plus'](use_BG=False,
                                                num_classes=10,
                                                final_layer_bias=False,)    

        checkpoint = torch.load(network_file, map_location=torch.device('cpu')) 
        net.load_state_dict(checkpoint)
        tools.device(net)
        
        gt, feats = [], []
        with torch.no_grad():
            for (x, y) in tqdm(data_loader, miniters=int(len(data_loader)/2), maxinterval=600, disable=not is_verbose):
                gt.extend(y.tolist())
                # assert False, f"{y.shape} {x.shape}"
                _, feat = net(tools.device(x))
                feats.extend(feat.tolist())
        gt, feats = np.array(gt), np.array(feats)
        max_val = np.max(np.abs(feats[gt!=-1,:]))
        xy_range = [-max_val, max_val]
        for c in range(10):
            feats_c = feats[gt==c,:]
            ax[i,j].scatter(feats_c[:,0], feats_c[:,1], label = c, s=1, color=CMAP[c])
        
        if idx == 0:
            ax[i,j].set_title(f'Initial')
        else:
            ax[i,j].set_title(f'{m}')
        ax[i,j].set_xlim(xy_range[0],xy_range[1])
        ax[i,j].set_ylim(xy_range[0],xy_range[1])
        ax[i,j].grid(True)
        print(f"{'Initial model' if idx==0 else m} Done! {idx+1}/{len(models)}")

    fig.suptitle('Feature space progress')
    fig.tight_layout()

    if save_plot:
        plt.savefig('./sample_feat_progress.png', bbox_inches="tight") 
        print("Figure saved! ./sample_feat_progress.png'")


if __name__ == '__main__':

    args = command_line_options()

    CMAP = mpl.color_sequences['tab10']

    num_classes = 10
    seed = 42
    gpu = args.gpu

    arch = args.arch
    scale = args.scale
    appr = args.approach[0]

    if gpu is not None and torch.cuda.is_available():
        tools.set_device_gpu(gpu)
    else:
        print("Running in CPU mode, might be slow")
        tools.set_device_cpu()

    deep_feat_viz_sample_progress(appr, seed, arch, scale, is_verbose=True, save_plot=True)