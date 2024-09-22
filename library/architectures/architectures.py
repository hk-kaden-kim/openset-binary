import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from collections import OrderedDict
import numpy as np
import math

from .. import tools

########################################################################
# Reference
# 
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

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

class ResNet_50(nn.Module):
    def __init__(self, feat_dim=-1, use_BG=False, num_classes=10, final_layer_bias=False, is_osovr=False, is_verbose=True):
        print("\n↓↓↓ Architecture setup ↓↓↓")
        print(f"{self.__class__.__name__} Architecture Loaded!")
        super(ResNet_50, self).__init__()
        resnet_base = models.resnet50(weights=None)
        fc_in_features = resnet_base.fc.in_features

        if use_BG: 
            num_classes += 1


        resnet_base.fc = nn.Linear(in_features=fc_in_features, 
                                   out_features=1000 if feat_dim == -1 else feat_dim)

        self.fc1 = resnet_base

        if is_osovr:
            self.fc2 = Linear_w_norm(in_features=1000 if feat_dim == -1 else feat_dim, 
                                     out_features=num_classes, bias=final_layer_bias)
            if is_verbose: print("Normalizing weights in the last linear layer.")
        else:
            self.fc2 = nn.Linear(in_features=1000 if feat_dim == -1 else feat_dim, 
                                out_features=num_classes, bias=final_layer_bias)
            
        if is_verbose:
            print(f"Set deep feature dimension to {1000 if feat_dim == -1 else feat_dim}")
            if final_layer_bias: print('Classifier has a bias term.')


    def forward(self, x):
        y = self.fc1(x) # Features
        x = self.fc2(y) # Logits
        return x, y

class LeNet_plus_plus(nn.Module):
    def __init__(self, use_BG=False, feat_dim=-1, num_classes=10, final_layer_bias=False, is_osovr=False, is_verbose=True):
        print("\n↓↓↓ Architecture setup ↓↓↓")
        print(f"{self.__class__.__name__} Architecture Loaded!")
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

        self.fc1 = nn.Linear(in_features=self.conv3_2.out_channels * 3 * 3,
                             out_features=2 if feat_dim == -1 else feat_dim)
        
        if is_osovr:
            self.fc2 = Linear_w_norm(in_features=2 if feat_dim == -1 else feat_dim, 
                                     out_features=num_classes, bias=final_layer_bias)
            if is_verbose: print("Normalizing weights in the last linear layer.")
        else:
            self.fc2 = nn.Linear(in_features=2 if feat_dim == -1 else feat_dim, 
                                out_features=num_classes, bias=final_layer_bias)

        self.prelu_act1 = nn.PReLU()
        self.prelu_act2 = nn.PReLU()
        self.prelu_act3 = nn.PReLU()

        if is_verbose: print(f"Set deep feature dimension to {2 if feat_dim == -1 else feat_dim}")
        if is_verbose and final_layer_bias: print('Classifier has a bias term.')

    def forward(self, x):
        x = self.prelu_act1(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        x = self.prelu_act2(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        x = self.prelu_act3(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)

        y = self.fc1(x) # Features
        x = self.fc2(y) # Logits

        # with torch.no_grad():
        #     scale = torch.norm(self.fc2.weight.data, p=2, dim=1)
        # x = (x / scale)
        
        return x, y
    
    def deep_feature_forward(self, y):
        return self.fc2(y)
    
class LeNet(nn.Module):
    def __init__(self, use_BG=False, feat_dim=-1, num_classes=10, final_layer_bias=False, is_osovr=False, is_verbose=True):
        print("\n↓↓↓ Architecture setup ↓↓↓")
        print(f"{self.__class__.__name__} Architecture Loaded!")
        super(LeNet, self).__init__()
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
            in_features=self.conv2.out_channels * 7 * 7, 
            out_features=500 if feat_dim == -1 else feat_dim, bias=True
        )

        if use_BG: num_classes += 1

        if is_osovr:
            self.fc2 = Linear_w_norm(in_features=500 if feat_dim == -1 else feat_dim, 
                                     out_features=num_classes, bias=final_layer_bias)
            if is_verbose: print("Normalizing weights in the last linear layer.")
        else:
            self.fc2 = nn.Linear(in_features=500 if feat_dim == -1 else feat_dim, 
                                out_features=num_classes, bias=final_layer_bias)
            

        self.relu_act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        if is_verbose:
            print(f"Set deep feature dimension to {500 if feat_dim == -1 else feat_dim}")
            if final_layer_bias: print('Classifier has a bias term.')
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

class Linear_w_norm(nn.Module):
    """reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
       reference2: <Additive Margin Softmax for Face Verification>
       reference3: <uccessfully and Efficiently Training Deep Multi-layer Perceptrons with Logistic Activation Function Simply Requires Initializing the Weights with an Appropriate Negative Mean>
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initial weight shape : (in_features, out_features)
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # nn.init.xavier_normal_(self.weight)
        
        # self.mean = max(-1, -8/in_features)
        # self.std = 0.1
        # nn.init.normal_(self.weight, mean=self.mean, std=self.std)

    def forward(self, feats:torch.Tensor):
        # print(self.weight)
        # with torch.no_grad():
        #     self.weight.data = F.normalize(self.weight.data, dim=0) # normalize weights in column-wise
        logits = torch.mm(feats, self.weight)
        with torch.no_grad():
            scale = 1/torch.norm(self.weight, p=2, dim=0)
        logits = logits * scale
        # print(self.weight)
        # print(logits)
        # assert False, "Terminated"
        return logits
    




# class ResNet_18(nn.Module):
#     def __init__(self, feat_dim=-1, use_BG=False, num_classes=10, final_layer_bias=False, is_verbose=True):
#         print("\n↓↓↓ Architecture setup ↓↓↓")
#         super(ResNet_18, self).__init__()
#         resnet_base = models.resnet18(weights=None)
#         fc_in_features = resnet_base.fc.in_features

#         if use_BG: num_classes += 1

#         resnet_base.fc = nn.Linear(in_features=fc_in_features, 
#                                    out_features=1024 if feat_dim == -1 else feat_dim)

#         self.fc1 = resnet_base
#         self.fc2 = nn.Linear(in_features=1024 if feat_dim == -1 else feat_dim
#                              , out_features=num_classes, bias=final_layer_bias)

#         if is_verbose:
#             print(f"Set deep feature dimension to {1024 if feat_dim == -1 else feat_dim}")
#             if final_layer_bias: print('Classifier has a bias term.')

#     def forward(self, x):
#         y = self.fc1(x) # Features
#         x = self.fc2(y) # Logits
#         return x, y
