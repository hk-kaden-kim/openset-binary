from library import architectures, tools, evals, dataset, losses
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pathlib
import os

batch_size = 1
num_workers = 5

def imagenet_save_img(imagenet_dataset, size=10, root='./'):

    df = imagenet_dataset.dataset
    
    kn_img_idx = df[df[1]!=-1].groupby(1).apply(lambda x: x.sample(1)).index
    kn_img_idx = np.random.choice([idx[1] for idx in kn_img_idx], size=size, replace=False)
    
    unkn_img_idx = df[df[1]==-1].index.to_numpy()
    unkn_img_idx = np.random.choice(unkn_img_idx, size=size, replace=False)

    for k_idx,u_idx in zip(kn_img_idx, unkn_img_idx):
        # print(k_idx, u_idx)
        image, label = imagenet_dataset[k_idx]
        image = np.transpose(image.numpy(), (1,2,0))
        img_name = df.iloc[k_idx,0].split('/')[-1].split('.')[0]
        img_name = f"{label}_"+img_name
        # print(label, image.shape)
        plt.figure(figsize=[6, 6])
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(root,f"{img_name}.png"))

        image, label = imagenet_dataset[u_idx]
        image = np.transpose(image.numpy(), (1,2,0))
        img_name = df.iloc[u_idx,0].split('/')[-1].split('.')[0]
        img_name = f"{label}_"+img_name
        # print(label, image.shape)
        plt.figure(figsize=[6, 6])
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(root,f"{img_name}.png"))


#########################################################
# LargeScale Dataset
#########################################################
# print("CHECK! - LargeScale Dataset")

# dataset_root = '/local/scratch/datasets/ImageNet/ILSVRC2012'
# protocol_root='./data/LargeScale'

# for protocol in range(1,4):

#     data = dataset.IMAGENET(dataset_root, protocol_root, protocol)
#     print(f"Protocol : {protocol}")

#     # Train / Val 
#     train_set_neg, val_set_neg, _ = data.get_train_set(has_background_class=False)
    
#     results_dir = pathlib.Path(f"./_images/{protocol}/train")
#     results_dir.mkdir(parents=True, exist_ok=True)
#     imagenet_save_img(train_set_neg, root=results_dir)
#     print(len(train_set_neg.dataset), '(All)', sum(train_set_neg.dataset[1]==-1), '(-1)')
    
#     results_dir = pathlib.Path(f"./_images/{protocol}/val")
#     results_dir.mkdir(parents=True, exist_ok=True)
#     imagenet_save_img(val_set_neg, root=results_dir)
#     print(len(val_set_neg.dataset), '(All)', sum(val_set_neg.dataset[1]==-1), '(-1)')

#     # Train / Val 

#     test_set_all, test_set_neg, test_set_unkn = data.get_test_set(has_background_class=False)

#     results_dir = pathlib.Path(f"./_images/{protocol}/test_neg")
#     results_dir.mkdir(parents=True, exist_ok=True)
#     imagenet_save_img(test_set_neg, root=results_dir)
#     print(len(test_set_neg.dataset), '(All)', sum(test_set_neg.dataset[1]==-1), '(-1)')
    
#     results_dir = pathlib.Path(f"./_images/{protocol}/test_unkn")
#     results_dir.mkdir(parents=True, exist_ok=True)
#     imagenet_save_img(test_set_unkn, root=results_dir)
#     print(len(test_set_unkn.dataset), '(All)', sum(test_set_unkn.dataset[1]==-1), '(-1)')






#########################################################
# SmallScale Dataset
#########################################################
print("CHECK! - SmallScale Dataset")
dataset_root = '/local/scratch/hkim'
data = dataset.EMNIST(dataset_root, convert_to_rgb=False)

data_mnist = data.train_mnist
data_letter = data.test_letters

kn_targets = [0,1,2,3,4,5,6,7,8,9]
neg_targets = [1,2,3,4,5,6,8,10,11,13,14]
unkn_targets = [16,17,18,19,20,21,22,23,24,25,26]

for targets in kn_targets:
    kn_idxs = [i for i, t in enumerate(data_mnist.targets) if t == targets]
    selected_idxs = np.random.choice(kn_idxs, 2)
    for idx in selected_idxs:
        x = data_mnist[idx][0]
        plt.imsave(f"./_images/mnist/mnist_{targets}_{idx}.png", x.squeeze().numpy(), cmap='Greys')

for targets in neg_targets:
    kn_idxs = [i for i, t in enumerate(data_letter.targets) if t == targets]
    selected_idxs = np.random.choice(kn_idxs, 2)
    for idx in selected_idxs:
        x = data_letter[idx][0]
        plt.imsave(f"./_images/letter/letter_neg_{targets}_{idx}.png", x.squeeze().numpy(), cmap='Greys')

for targets in unkn_targets:
    kn_idxs = [i for i, t in enumerate(data_letter.targets) if t == targets]
    selected_idxs = np.random.choice(kn_idxs, 2)
    for idx in selected_idxs:
        x = data_letter[idx][0]
        plt.imsave(f"./_images/letter/letter_unkn_{targets}_{idx}.png", x.squeeze().numpy(), cmap='Greys')


# training_data, val_data, num_classes = data.get_train_set(has_background_class=False,
#                                                           size_train_negatives=20000)


# gt_labels = dataset.get_gt_labels(training_data, is_verbose=True)
# stats = np.unique(gt_labels.cpu(), return_counts=True)
# evals.print_table(stats[0], stats[1])

# gt_labels = dataset.get_gt_labels(val_data, is_verbose=True)
# stats = np.unique(gt_labels.cpu(), return_counts=True)
# evals.print_table(stats[0], stats[1])

# train_data_loader = torch.utils.data.DataLoader(
#     training_data,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     pin_memory=True
# )


# pos = False
# neg = False

# for x, y in train_data_loader:

#     if not pos and y == 7:
#         print('Get 7', y, x.shape)
#         plt.imsave("7.png", x.squeeze().numpy(), cmap='Greys')
#         pos = True

#     if not neg and y == -1:
#         print('Get Letter', y, x.shape)
#         plt.imsave("letter.png", x.squeeze().numpy(), cmap='Greys')
#         neg = True

#     if pos and neg:
#         print('Get All!')
#         break