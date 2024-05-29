import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, ConcatDataset

from sklearn.model_selection import train_test_split

def transpose(x):
    """Used for correcting rotation of EMNIST Letters"""
    return x.transpose(2,1)

class EMNIST():
    """...

    Parameters:

    ... : ...

    """
    
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.train_mnist = torchvision.datasets.EMNIST(
                        root=self.dataset_root,
                        train=True,
                        download=False, 
                        split="mnist",
                        transform=transforms.Compose([transforms.ToTensor(), transpose])
                    )
        self.test_mnist = torchvision.datasets.EMNIST(
                        root=self.dataset_root,
                        train=False,
                        download=False,
                        split="mnist",
                        transform=transforms.Compose([transforms.ToTensor(), transpose])
                    )
        self.train_letters = torchvision.datasets.EMNIST(
                        root=dataset_root,
                        train=True,
                        download=False,
                        split='letters',
                        transform=transforms.Compose([transforms.ToTensor(), transpose])
                    )
        self.test_letters = torchvision.datasets.EMNIST(
                        root=dataset_root,
                        train=False,
                        download=False,
                        split='letters',
                        transform=transforms.Compose([transforms.ToTensor(), transpose])
                    )
        
    def get_train_set(self, split_ratio:float, include_negatives=False, has_background_class=False):

        mnist_idxs = [i for i, _ in enumerate(self.train_mnist.targets)]
        tr_mnist_idxs, val_mnist_idxs = train_test_split(mnist_idxs, train_size=split_ratio)
        tr_mnist, val_mnist = Subset(self.train_mnist, tr_mnist_idxs), Subset(self.train_mnist, val_mnist_idxs)

        letters_targets = [1,2,3,4,5,6,8,10,11,13,14]
        letters_idxs = [i for i, t in enumerate(self.train_letters.targets) if t in letters_targets]
        tr_letters_idxs, val_letters_idxs = train_test_split(letters_idxs, train_size=split_ratio)
        tr_letters, val_letters = Subset(self.train_letters, tr_letters_idxs), Subset(self.train_letters, val_letters_idxs)
        tr_letters, val_letters = UnknownDataset(tr_letters,has_background_class), UnknownDataset(val_letters,has_background_class)

        if include_negatives:
            train_emnist, val_emnist = ConcatDataset([tr_mnist, tr_letters]), ConcatDataset([val_mnist, val_letters])
        else:
            train_emnist, val_emnist = ConcatDataset([tr_mnist]), ConcatDataset([val_mnist])

        return (train_emnist, val_emnist)

    def get_test_set(self, has_background_class=False):

        mnist_idxs = [i for i, _ in enumerate(self.test_mnist.targets)]
        test_mnist = Subset(self.test_mnist, mnist_idxs)

        letters_targets = [1,2,3,4,5,6,8,10,11,13,14]
        letters_idxs = [i for i, t in enumerate(self.test_letters.targets) if t in letters_targets]
        test_neg_letters = Subset(self.test_letters, letters_idxs)
        test_neg_letters = UnknownDataset(test_neg_letters,has_background_class)

        letters_targets = [16,17,18,19,20,21,22,23,24,25,26]
        letters_idxs = [i for i, t in enumerate(self.test_letters.targets) if t in letters_targets]
        test_unkn_letters = Subset(self.test_letters, letters_idxs)
        test_unkn_letters = UnknownDataset(test_unkn_letters,has_background_class)

        test_kn_neg = ConcatDataset([test_mnist, test_neg_letters])
        test_kn_unkn = ConcatDataset([test_mnist, test_unkn_letters])
        test_all = ConcatDataset([test_mnist, test_neg_letters, test_unkn_letters])

        return test_all, test_kn_neg, test_kn_unkn

class UnknownDataset(torch.utils.data.dataset.Subset):

    def __init__(self, subset, has_background_class):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.has_background_class = has_background_class

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]][0], 10 if self.has_background_class else -1

    def check(self, index):
        return index, int(self.dataset.targets[self.indices[index]]), 10 if self.has_background_class else -1

