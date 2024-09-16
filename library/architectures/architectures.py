import torch.nn as nn
import torch
from torchvision import models
from collections import OrderedDict
import numpy as np
from .. import tools

def get_init_weights(num_points, dimension):

    # Option 1. Init weight on the axis
    # w_lookup = torch.concat((torch.eye(dimension),-1 * torch.eye(dimension)))
    # init_w = torch.zeros((num_points,dimension))
    # for i in range(num_points):
    #     k = i % w_lookup.shape[0]
    #     init_w[i,:] = w_lookup[k,:]

    # Option 2. Init weight to one
    # init_w = torch.empty(num_points, dimension)
    # nn.init.ones_(init_w)

    # Option 3. Xavier uniform
    init_w = torch.empty(num_points, dimension)
    nn.init.xavier_uniform_(init_w, gain=nn.init.calculate_gain('relu'))

    return tools.device(init_w)


class ResNet_18(nn.Module):
    def __init__(self, force_fc_dim=1000, use_BG=False, num_classes=10, init_weights=False, final_layer_bias=False, is_verbose=False):
        super(ResNet_18, self).__init__()
        resnet_base = models.resnet18(weights=None)
        fc_in_features = resnet_base.fc.in_features

        if use_BG: 
            num_classes += 1

        if force_fc_dim != -1:
            fc_layer_dim=force_fc_dim
        else:
            fc_layer_dim = num_classes

        resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.fc1 = resnet_base
        if final_layer_bias:
            print('Classifier has a bias term.')
        self.fc2 = nn.Linear(in_features=fc_layer_dim, out_features=num_classes, bias=final_layer_bias)
        if init_weights:
            fc2_init_weights = get_init_weights(num_classes, fc_layer_dim)
            self.fc2.weight = torch.nn.Parameter(fc2_init_weights)
            print(f"Initialize weights on the last layer!\nInitial value [:3]\n{self.fc2.weight[:3]}")

    def forward(self, x):
        y = self.fc1(x) # Features
        x = self.fc2(y) # Logits
        return x, y

class ResNet_50(nn.Module):
    def __init__(self, force_fc_dim=1000, use_BG=False, num_classes=10, init_weights=False, final_layer_bias=False, is_verbose=False):
        super(ResNet_50, self).__init__()
        resnet_base = models.resnet50(weights=None)
        fc_in_features = resnet_base.fc.in_features

        if use_BG: 
            num_classes += 1

        if force_fc_dim != -1:
            fc_layer_dim=force_fc_dim
            # print(f"Feature Space Dimension: {fc_layer_dim}\n")
        else:
            fc_layer_dim = num_classes

        resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.fc1 = resnet_base
        if final_layer_bias:
            print('Classifier has a bias term.')
        self.fc2 = nn.Linear(in_features=fc_layer_dim, out_features=num_classes, bias=final_layer_bias)
        if init_weights:
            fc2_init_weights = get_init_weights(num_classes, fc_layer_dim)
            self.fc2.weight = torch.nn.Parameter(fc2_init_weights)
            print(f"Initialize weights on the last layer!\nInitial value [:3]\n{self.fc2.weight[:3]}")

    def forward(self, x):
        y = self.fc1(x) # Features
        x = self.fc2(y) # Logits
        return x, y

########################################################################
# Reference
# 
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

class LeNet_plus_plus(nn.Module):
    def __init__(self, use_BG=False, num_classes=10, init_weights=False, final_layer_bias=False, is_verbose=False):
        super(LeNet_plus_plus, self).__init__()
        self.conv1_1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=self.conv1_1.out_channels,
            out_channels=32,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.batch_norm1 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2_1 = nn.Conv2d(
            in_channels=self.conv1_2.out_channels,
            out_channels=64,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=self.conv2_1.out_channels,
            out_channels=64,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.batch_norm2 = nn.BatchNorm2d(self.conv2_2.out_channels)
        self.conv3_1 = nn.Conv2d(
            in_channels=self.conv2_2.out_channels,
            out_channels=128,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.conv3_2 = nn.Conv2d(
            in_channels=self.conv3_1.out_channels,
            out_channels=128,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.batch_norm3 = nn.BatchNorm2d(self.conv3_2.out_channels)
        
        if use_BG: num_classes += 1

        self.fc1 = nn.Linear(in_features=self.conv3_2.out_channels * 3 * 3, out_features=2)
        if final_layer_bias:
            print('Classifier has a bias term.')
        self.fc2 = nn.Linear(in_features=2, out_features=num_classes, bias=final_layer_bias)
        if init_weights:
            fc2_init_weights = get_init_weights(num_classes, 2)
            self.fc2.weight = torch.nn.Parameter(fc2_init_weights)
            print(f"Initialize weights on the last layer!\nInitial value [:3]\n{self.fc2.weight[:3]}")

        self.prelu_act1 = nn.PReLU()
        self.prelu_act2 = nn.PReLU()
        self.prelu_act3 = nn.PReLU()

    def forward(self, x):
        x = self.prelu_act1(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        x = self.prelu_act2(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        x = self.prelu_act3(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)

        y = self.fc1(x) # Features
        x = self.fc2(y) # Logits

        return x, y
    
    def deep_feature_forward(self, y):
        return self.fc2(y)
    
class LeNet(nn.Module):
    def __init__(self, use_BG=False, num_classes=10, init_weights=False, final_layer_bias=False, is_verbose=True):
        super(LeNet, self).__init__()
        deep_feats = 500        # 500
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=20, 
            kernel_size=(5, 5), 
            stride=1, padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=50,
            kernel_size=(5, 5),
            stride=1, padding=2,
        )
        self.fc1 = nn.Linear(
            in_features=self.conv2.out_channels * 7 * 7, out_features=deep_feats, bias=True
        )
        if final_layer_bias:
            print('Classifier has a bias term.')
        if use_BG:
            self.fc2 = nn.Linear(
                in_features=deep_feats, out_features=num_classes + 1, bias=final_layer_bias
            )
        else:
            self.fc2 = nn.Linear(in_features=deep_feats, out_features=num_classes, bias=final_layer_bias)
        if init_weights:
            fc2_init_weights = get_init_weights(num_classes, deep_feats)
            self.fc2.weight = torch.nn.Parameter(fc2_init_weights)
            print(f"Initialize weights on the last layer!\nInitial value [:3]\n{self.fc2.weight[:3]}")

        self.relu_act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        if is_verbose:
            print(
                f"{' Model Architecture '.center(90, '#')}\n{self}\n{' Model Architecture End '.center(90, '#')}"
            )

    def forward(self, x):
        x = self.pool(self.relu_act(self.conv1(x)))
        x = self.pool(self.relu_act(self.conv2(x)))
        x = x.view(-1, self.conv2.out_channels * 7 * 7)

        y = self.fc1(x) # Features
        x = self.fc2(y) # Logits

        return x, y
