import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, ConcatDataset

from sklearn.model_selection import train_test_split

import pathlib
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

from tqdm import tqdm
from .. import tools


def transpose(x):
    """Used for correcting rotation of EMNIST"""
    return x.transpose(2,1)

def get_gt_labels(dataset, batch_size=1024, gpu=None, is_verbose=False):

    if is_verbose:
        print(f"Get Ground Truth Labels.")
    # if gpu is not None and torch.cuda.is_available():
    #     tools.set_device_gpu(gpu)
    # else:
    #     if is_verbose:
    #         print("Running in CPU mode, might be slow")
    #     tools.set_device_cpu()

    # gt_labels = tools.device(torch.Tensor())
    gt_labels = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (_, y) in tqdm(data_loader, miniters=int(len(data_loader)/5), maxinterval=600, disable=not is_verbose):
            y = tools.device(y)
            gt_labels.extend(y.tolist())
            # gt_labels = torch.cat((gt_labels, y))

    gt_labels = tools.device(torch.Tensor(gt_labels))

    return gt_labels

def calc_class_weight(dataset, batch_size=1024, gpu=None, is_verbose=False):
    
    gt_labels = get_gt_labels(dataset, batch_size, gpu, is_verbose)
    total_num = len(gt_labels)
    u_labels, cnts = gt_labels.unique(return_counts=True)
    u_labels = u_labels.to(torch.int)

    c_w = []
    for l in u_labels:
        c_w.append(cnts[l]/total_num)

    return tools.device(torch.Tensor(c_w))


class EMNIST():
    """...

    Parameters:

    ... : ...

    """
    
    def __init__(self, dataset_root, split_ratio=0.8, label_filter=[-1], seed=42, convert_to_rgb=False):
        print("\n↓↓↓ Dataset setup ↓↓↓")
        print(f"{self.__class__.__name__} Dataset Loaded!")
        self.dataset_root = dataset_root
        self.split_ratio = split_ratio
        self.seed = seed
        self.label_filter = label_filter

        data_transform = [transforms.ToTensor(), transpose]
        if convert_to_rgb:
            data_transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        self.train_mnist = torchvision.datasets.EMNIST(
                        root=self.dataset_root,
                        train=True,         # TRAIN
                        download=False, 
                        split="mnist",
                        transform=transforms.Compose(data_transform)
                    )
        self.test_mnist = torchvision.datasets.EMNIST(
                        root=self.dataset_root,
                        train=False,           # TEST
                        download=False,
                        split="mnist",
                        transform=transforms.Compose(data_transform)
                    )
        self.train_letters = torchvision.datasets.EMNIST(
                        root=dataset_root,
                        train=True,             # TRAIN
                        download=False,
                        split='letters',
                        transform=transforms.Compose(data_transform)
                    )
        self.test_letters = torchvision.datasets.EMNIST(
                        root=dataset_root,
                        train=False,            # TEST
                        download=False,
                        split='letters',
                        transform=transforms.Compose(data_transform)
                    )
        
    def get_train_set(self, size_train_negatives=-1, has_background_class=False, is_verbose=True):
        
        # Get MNIST for Known samples
        if self.label_filter[0] == -1:
            mnist_idxs = [i for i, _ in enumerate(self.train_mnist.targets)]
        else:
            mnist_idxs = [i for i, t in enumerate(self.train_mnist.targets) if t in self.label_filter]
            if is_verbose: print(f"Filtered Target Label : {self.label_filter}")
        tr_mnist_idxs, val_mnist_idxs = train_test_split(mnist_idxs, train_size=self.split_ratio, random_state=self.seed)
        tr_mnist, val_mnist = Subset(self.train_mnist, tr_mnist_idxs), Subset(self.train_mnist, val_mnist_idxs)

        # Get Letters for Neg Samples
        letters_targets = [1,2,3,4,5,6,8,10,11,13,14]
        letters_idxs = [i for i, t in enumerate(self.train_letters.targets) if t in letters_targets]
        tr_letters_idxs, val_letters_idxs = train_test_split(letters_idxs, train_size=self.split_ratio, random_state=self.seed)

        # Negatives in training
        if size_train_negatives == 0:
            if is_verbose:
                print(f"# of negatives for training: {size_train_negatives}")
            val_letters = Subset(self.train_letters, val_letters_idxs)
            val_letters = EmnistUnknownDataset(val_letters,has_background_class)
            train_emnist = ConcatDataset([tr_mnist])
            val_emnist = ConcatDataset([val_mnist, val_letters])
        else:
            # Reduce the size of negatives in training set
            if size_train_negatives > 0:
                assert len(tr_letters_idxs) > size_train_negatives, f"The required size of train negatives ({size_train_negatives}) is too big. It should be smaller than  {len(tr_letters_idxs)}, which is {len(letters_idxs)} x {self.split_ratio}."
                tr_letters_idxs = list(np.sort(np.random.choice(tr_letters_idxs, size_train_negatives)))
                if is_verbose:
                    print(f"# of negatives for training: {size_train_negatives} {len(tr_letters_idxs)}")
            elif size_train_negatives == -1:
                if is_verbose:
                    print(f"# of negatives for training: -1 >> ALL {int(len(tr_letters_idxs))}")
            tr_letters, val_letters = Subset(self.train_letters, tr_letters_idxs), Subset(self.train_letters, val_letters_idxs)
            tr_letters, val_letters = EmnistUnknownDataset(tr_letters,has_background_class), EmnistUnknownDataset(val_letters,has_background_class)
            train_emnist = ConcatDataset([tr_mnist, tr_letters])
            val_emnist = ConcatDataset([val_mnist, val_letters])

        return (train_emnist, val_emnist, 10 if self.label_filter[0] == -1 else len(self.label_filter))

    def get_test_set(self, has_background_class=False, is_verbose=True):

        if self.label_filter[0] == -1:
            mnist_idxs = [i for i, _ in enumerate(self.test_mnist.targets)]
        else:
            mnist_idxs = [i for i, t in enumerate(self.test_mnist.targets) if t in self.label_filter]
            if is_verbose: print(f"Filtered Target Label : {self.label_filter}")
        test_mnist = Subset(self.test_mnist, mnist_idxs)

        letters_targets = [1,2,3,4,5,6,8,10,11,13,14]
        letters_idxs = [i for i, t in enumerate(self.test_letters.targets) if t in letters_targets]
        test_neg_letters = Subset(self.test_letters, letters_idxs)
        test_neg_letters = EmnistUnknownDataset(test_neg_letters,has_background_class)

        letters_targets = [16,17,18,19,20,21,22,23,24,25,26]
        letters_idxs = [i for i, t in enumerate(self.test_letters.targets) if t in letters_targets]
        test_unkn_letters = Subset(self.test_letters, letters_idxs)
        test_unkn_letters = EmnistUnknownDataset(test_unkn_letters,has_background_class)

        test_kn_neg = ConcatDataset([test_mnist, test_neg_letters])
        test_kn_unkn = ConcatDataset([test_mnist, test_unkn_letters])
        test_all = ConcatDataset([test_mnist, test_neg_letters, test_unkn_letters])

        return test_all, test_kn_neg, test_kn_unkn

class EmnistUnknownDataset(torch.utils.data.dataset.Subset):

    def __init__(self, subset, has_background_class):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.has_background_class = has_background_class

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]][0], 10 if self.has_background_class else -1

    def check_len(self, index):
        return index, int(self.dataset.targets[self.indices[index]]), 10 if self.has_background_class else -1

    def check_stats(self):
        label = []
        for idx in self.indices:
            label.append(self.dataset[idx][1])
        label = np.array(label)
        return np.unique(label, return_counts=True)

class IMAGENET():
    def __init__(self, dataset_root, protocol_root, protocol=1, is_verbose=False):
        print("\n↓↓↓ Dataset setup ↓↓↓")
        print(f"{self.__class__.__name__} Dataset Loaded!")
        if is_verbose:
            print(f"Protocol: {protocol}")

        # Set image transformations
        self.train_data_transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()])

        self.val_data_transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
        
        # create datasets
        self.train_file = Path(os.path.join(protocol_root, f'protocols/p{protocol}_train.csv'))
        self.val_file = Path(os.path.join(protocol_root, f'protocols/p{protocol}_val.csv'))
        self.test_file = Path(os.path.join(protocol_root, f'protocols/p{protocol}_test.csv'))

        self.dataset_root = dataset_root
        if not self.train_file.exists():
            raise FileNotFoundError(f"ImageNet Train Protocol is not exist at {self.train_file}")
        if not self.val_file.exists():
            raise FileNotFoundError(f"ImageNet Train Protocol is not exist at {self.val_file}")
        if not self.test_file.exists():
            raise FileNotFoundError(f"ImageNet Train Protocol is not exist at {self.test_file}")

    def get_train_set(self, size_train_negatives=-1, has_background_class=False, is_verbose=False):

        train_ds = ImagenetDataset(
                csv_file=self.train_file,
                imagenet_path=self.dataset_root,
                transform=self.train_data_transform
            )
        
        val_ds = ImagenetDataset(
                csv_file=self.val_file,
                imagenet_path=self.dataset_root,
                transform=self.val_data_transform
            )


        # If we need to reduce the size of negatives in training set
        if size_train_negatives == 0:
            if is_verbose: print(f"# of negatives for training: {size_train_negatives}")
            train_ds.remove_negative_label()
        elif size_train_negatives == -1:
            if is_verbose: print(f"# of negatives for training: {train_ds.get_negatives_size()}")
        else:
            assert False, f"Not avilable to set the size of Negatives in Large-scale training dataset"

        return (train_ds, val_ds, train_ds.label_count)

    def get_test_set(self, has_background_class=False, is_verbose=False):

        # Initialize
        # Known + Negative + Unknown
        test_dataset = ImagenetDataset(
                csv_file=self.test_file,
                imagenet_path=self.dataset_root,
                transform=self.val_data_transform
            )   
        # Known + Negative
        test_neg_dataset = ImagenetDataset(
                csv_file=self.test_file,
                imagenet_path=self.dataset_root,
                transform=self.val_data_transform
            )   
        # Known + Unknown
        test_unkn_dataset = ImagenetDataset(
                csv_file=self.test_file,
                imagenet_path=self.dataset_root,
                transform=self.val_data_transform
            )   
        

        test_dataset.dataset[1] = test_dataset.dataset[1].replace(-2, -1) # Replace unknown(-2) to -1

        test_neg_dataset.dataset = test_neg_dataset.dataset[test_neg_dataset.dataset[1] > -2] # filter out: unknown (-2)
        test_neg_dataset.dataset.reset_index(inplace=True, drop=True)
        
        test_unkn_dataset.dataset = test_unkn_dataset.dataset[test_unkn_dataset.dataset[1] != -1]   # filter out: negative (-1)
        test_unkn_dataset.dataset[1] = test_unkn_dataset.dataset[1].replace(-2, -1)
        test_unkn_dataset.dataset.reset_index(inplace=True, drop=True)

        return test_dataset, test_neg_dataset, test_unkn_dataset


########################################################################
# Reference
# 
# Author: UZH AIML Group
# Date: 2024
# Availability: https://github.com/AIML-IfI/openset-imagenet
########################################################################

class ImagenetDataset(torch.utils.data.dataset.Dataset):
    """ Imagenet Dataset. """

    def __init__(self, csv_file, imagenet_path, transform=None):
        """ Constructs an Imagenet Dataset from a CSV file. The file should list the path to the
        images and the corresponding label. For example:
        val/n02100583/ILSVRC2012_val_00013430.JPEG,   0

        Args:
            csv_file(Path): Path to the csv file with image paths and labels.
            imagenet_path(Path): Home directory of the Imagenet dataset.
            transform(torchvision.transforms): Transforms to apply to the images.
        """
        self.dataset = pd.read_csv(csv_file, header=None)
        self.imagenet_path = Path(imagenet_path)
        self.transform = transform
        self.label_count = len(self.dataset[self.dataset[1]>=0][1].unique())
        self.unique_classes = np.sort(self.dataset[1].unique())

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.dataset)

    def __getitem__(self, index):
        """ Returns a tuple (image, label) of the dataset at the given index. If available, it
        applies the defined transform to the image. Images are converted to RGB format.

        Args:
            index(int): Image index

        Returns:
            image, label: (image tensor, label tensor)
        """
        if torch.is_tensor(index):
            index = index.tolist()

        jpeg_path, label = self.dataset.iloc[index]
        image = Image.open(self.imagenet_path / jpeg_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # convert int label to tensor
        label = torch.as_tensor(int(label), dtype=torch.int64)
        return image, label

    def has_negatives(self):
        """ Returns true if the dataset contains negative samples."""
        return -1 in self.unique_classes

    def replace_negative_label(self, update_label_cnt=False):
        """ Replaces negative label (-1) to biggest_label + 1. This is required if the loss function
        is BGsoftmax. Updates the array of unique labels.
        """
        biggest_label = self.label_count
        self.dataset[1] = self.dataset[1].replace(-1, biggest_label)
        self.unique_classes[self.unique_classes == -1] = biggest_label
        self.unique_classes.sort()
        if update_label_cnt:
            self.label_count += 1

    def replace_unknown_label(self, update_label_cnt=False):
        """ Replaces negative label (-2) to biggest_label + 1. This is required if the loss function
        is BGsoftmax. Updates the array of unique labels.
        """
        biggest_label = self.label_count
        self.dataset[1] = self.dataset[1].replace(-2, biggest_label)
        self.unique_classes[self.unique_classes == -1] = biggest_label
        self.unique_classes.sort()
        if update_label_cnt:
            self.label_count += 1

    def remove_negative_label(self):
        """ Removes all negative labels (<0) from the dataset. This is required for training with plain softmax"""
        self.dataset = self.dataset.drop(self.dataset[self.dataset[1] < 0].index)
        self.unique_classes = np.sort(self.dataset[1].unique())
        self.label_count = len(self.dataset[1].unique())

    def get_negatives_size(self):
        return sum(self.dataset[1] < 0)

    def calculate_class_weights(self):
        """ Calculates the class weights based on sample counts.

        Returns:
            class_weights: Tensor with weight for every class.
        """
        counts = self.dataset.groupby(1).count().to_numpy()
        class_weights = (len(self.dataset) / (counts * self.label_count))
        return torch.from_numpy(class_weights).float().squeeze()