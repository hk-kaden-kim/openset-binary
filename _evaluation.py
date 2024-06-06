import torch
from torch.nn import functional as F
import numpy
import os

from library import architectures, tools, evals, dataset

import matplotlib
matplotlib.rcParams["font.size"] = 18
from matplotlib import pyplot, patches
from matplotlib.backends.backend_pdf import PdfPages


labels={
  "SoftMax" : "Plain SoftMax",
  "Garbage" : "Garbage Class",
  "EOS" : "Entropic Open-Set",
  "Objectosphere" : "Objectosphere",
  "MultiBinary" : "Multiple Binary Classifiers",
}



class Arguments:
    def __init__(self, dataset_root, plot, approaches='SoftMax', dataset='SmallScale', arch="LeNet_plus_plus"):
        self.approaches = approaches
        self.arch = arch
        self.dataset = dataset
        self.dataset_root = dataset_root
        self.plot = plot





args = Arguments(approaches = ["SoftMax", "Garbage", "EOS", "Objectosphere", "MultiBinary"],
                 arch = 'LeNet_plus_plus',
                 dataset = 'SmallScale',
                 dataset_root = '/local/scratch/hkim',
                 plot='Test_Evaluate.pdf')




print('-----------------------------------------------')
print('Load model')
print('-----------------------------------------------')

# networks
networks = {
    which: evals.load_network(args, which) for which in args.approaches
}




print('-----------------------------------------------')
print('Load data')
print('-----------------------------------------------')
# dataset
if args.dataset == 'SmallScale':
    data = dataset.EMNIST(args.dataset_root)
else:
    data = ... # LargeScale

train_set, val_set = data.get_train_set(split_ratio=0.8, include_negatives=False, has_background_class=False)
train_set_neg, val_set_neg = data.get_train_set(split_ratio=0.8, include_negatives=True, has_background_class=False)
train_set_neg_bg, val_set_neg_bg = data.get_train_set(split_ratio=0.8, include_negatives=True, has_background_class=True)

test_set_all, test_set_neg, test_set_unkn = data.get_test_set(has_background_class=False)
test_set_all_bg, test_set_neg_bg, test_set_unkn_bg = data.get_test_set(has_background_class=True)




print('-----------------------------------------------')
print('Predict')
print('-----------------------------------------------')

# Select: "SoftMax", "Garbage", "EOS", "Objectosphere", "MultiBinary"
which = "SoftMax"
net = networks[which]

# Predicts
train_gt, train_logits, train_feats = evals.extract(train_set_neg, net)
test_neg_gt, test_neg_logits, test_neg_feats = evals.extract(test_set_neg, net)
test_unkn_gt, test_unkn_logits, test_unkn_feats = evals.extract(test_set_unkn, net)

# Calculate Probs
if args.approaches == "MultiBinary":
    train_probs = F.sigmoid(torch.tensor(train_logits)).detach().numpy()
    test_neg_probs = F.sigmoid(torch.tensor(test_neg_logits)).detach().numpy()
    test_unkn_probs  = F.sigmoid(torch.tensor(test_unkn_logits )).detach().numpy()
else:
    train_probs = F.softmax(torch.tensor(train_logits), dim=1).detach().numpy()
    test_neg_probs = F.softmax(torch.tensor(test_neg_logits), dim=1).detach().numpy()
    test_unkn_probs  = F.softmax(torch.tensor(test_unkn_logits ), dim=1).detach().numpy()

# remove the labels for the unknown class in case of Garbage Class
if args.approaches == "Garbage":
    test_neg_probs = test_neg_probs[:,:-1]
    test_unkn_probs = test_unkn_probs[:,:-1]
    unkn_gt_label = 10  # Change the lable of unkn gt
else:
    unkn_gt_label = -1




def get_probs(pnts, which, net):

    pnts = torch.tensor(pnts).float()
    result = net.deep_feature_forward(pnts)
    if which == 'MultiBinary':
        probs = F.sigmoid(result).detach()
    else:
        probs = F.softmax(result, dim=1).detach()
    probs = torch.max(probs, dim=1).values
    
    return probs





print('-----------------------------------------------')
print('Plot train dataset')
print('-----------------------------------------------')

data = train_feats[train_gt != unkn_gt_label]
data_labels = train_gt[train_gt != unkn_gt_label]
neg_features = train_feats[train_gt == unkn_gt_label]

tools.viz.plotter_2D(data, data_labels, neg_features=None,
               final=True, heat_map=False, file_name='1_train_{}.{}')

tools.viz.plotter_2D(data, data_labels, neg_features=None,
               final=True, heat_map=True, file_name='2_train_{}_heat.{}', prob_function=get_probs, which=which, net=net)

tools.viz.plotter_2D(data, data_labels, neg_features=neg_features,
               final=True, heat_map=False, file_name='3_train_{}_neg.{}')





print('-----------------------------------------------')
print('Plot test(neg) dataset')
print('-----------------------------------------------')

data = test_neg_feats[test_neg_gt != unkn_gt_label]
data_labels = test_neg_gt[test_neg_gt != unkn_gt_label]
neg_features = test_neg_feats[test_neg_gt == unkn_gt_label]

tools.viz.plotter_2D(data, data_labels, neg_features=None,
               final=True, heat_map=False, file_name='1_test_{}.{}')

tools.viz.plotter_2D(data, data_labels, neg_features=None,
               final=True, heat_map=True, file_name='2_test_{}_heat.{}', prob_function=get_probs, which=which, net=net)

tools.viz.plotter_2D(data, data_labels, neg_features=neg_features,
               final=True, heat_map=False, file_name='3_test_{}_neg.{}')



print('-----------------------------------------------')
print('Plot test(unkn) dataset')
print('-----------------------------------------------')


data = test_unkn_feats[test_neg_gt != unkn_gt_label]
data_labels = test_unkn_gt[test_neg_gt != unkn_gt_label]
neg_features = test_unkn_feats[test_neg_gt == unkn_gt_label]

# tools.viz.plotter_2D(data, data_labels, neg_features=None,
#                final=True, heat_map=False, file_name=None)

# tools.viz.plotter_2D(data, data_labels, neg_features=None,
#                final=True, heat_map=True, file_name=None, prob_function=get_probs, which=which, net=net)

tools.viz.plotter_2D(data, data_labels, neg_features=neg_features,
               final=True, heat_map=False, file_name='3_test_{}_unkn.{}')