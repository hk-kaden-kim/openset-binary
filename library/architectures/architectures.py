import torch.nn as nn
from torchvision import models
from collections import OrderedDict

class ResNet_18(nn.Module):
    def __init__(self, force_fc_dim=-1, use_BG=False, num_classes=10, final_layer_bias=True):
        super(ResNet_18, self).__init__()
        resnet_base = models.resnet18(weights=None)
        fc_in_features = resnet_base.fc.in_features

        if use_BG: 
            num_classes += 1

        if force_fc_dim != -1:
            fc_layer_dim=force_fc_dim
            print(f"Feature Space Dimension: {fc_layer_dim}\n")
        else:
            fc_layer_dim = num_classes

        resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.fc1 = resnet_base
        self.fc2 = nn.Linear(in_features=fc_layer_dim, out_features=num_classes, bias=final_layer_bias)
        
    def forward(self, x):
        y = self.fc1(x) # Features
        x = self.fc2(y) # Logits
        return x, y

class ResNet_50(nn.Module):
    def __init__(self, force_fc_dim=-1, use_BG=False, num_classes=10, final_layer_bias=True):
        super(ResNet_50, self).__init__()
        resnet_base = models.resnet50(weights=None)
        fc_in_features = resnet_base.fc.in_features

        if use_BG: 
            num_classes += 1

        if force_fc_dim != -1:
            fc_layer_dim=force_fc_dim
            print(f"Feature Space Dimension: {fc_layer_dim}\n")
        else:
            fc_layer_dim = num_classes

        resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.fc1 = resnet_base
        self.fc2 = nn.Linear(in_features=fc_layer_dim, out_features=num_classes, bias=final_layer_bias)
                
    def forward(self, x):
        y = self.fc1(x) # Features
        x = self.fc2(y) # Logits
        return x, y

def checkpoint_sync(checkpoint:OrderedDict):
    
    new_checkpoint = OrderedDict()
    for key, value in checkpoint.items():
        if 'resnet_base' in key:
            new_checkpoint.update({key.replace('resnet_base','fc1'):value})
        else:
            new_checkpoint.update({key.replace('logits','fc2'):value})
    
    return new_checkpoint

########################################################################
# Reference
# 
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

class LeNet_plus_plus(nn.Module):
    def __init__(self, force_fc_dim=-1, use_BG=False, num_classes=10, final_layer_bias=True):
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
        
        if use_BG: 
            num_classes += 1

        if force_fc_dim != -1:
            fc_layer_dim=force_fc_dim
            print(f"Feature Space Dimension: {fc_layer_dim}\n")
        else:
            fc_layer_dim = num_classes

        self.fc1 = nn.Linear(
            in_features=self.conv3_2.out_channels * 3 * 3, out_features=fc_layer_dim, bias=True
        )

        self.fc2 = nn.Linear(in_features=fc_layer_dim, out_features=num_classes, bias=final_layer_bias)

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


########################################################################
# LeNet++ Test for the effect of Feature Space Dimension
########################################################################

class LeNet_plus_plus_3(nn.Module):
    def __init__(self, use_classification_layer=True, use_BG=False, num_classes=10, final_layer_bias=True):
        super(LeNet_plus_plus_3, self).__init__()
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
        self.fc1 = nn.Linear(
            in_features=self.conv3_2.out_channels * 3 * 3, out_features=3, bias=True
        )
        if use_classification_layer:
            if use_BG:
                self.fc2 = nn.Linear(
                    in_features=3, out_features=num_classes + 1, bias=final_layer_bias
                )
            else:
                self.fc2 = nn.Linear(in_features=3, out_features=num_classes, bias=final_layer_bias)
        self.use_classification_layer = use_classification_layer
        self.prelu_act1 = nn.PReLU()
        self.prelu_act2 = nn.PReLU()
        self.prelu_act3 = nn.PReLU()

    def forward(self, x):
        x = self.prelu_act1(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        x = self.prelu_act2(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        x = self.prelu_act3(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)

        y = self.fc1(x) # Features
        if self.use_classification_layer:
            x = self.fc2(y) # Logits
            return x, y
        return y
    
    def deep_feature_forward(self, y):
        return self.fc2(y)


class LeNet_plus_plus_5(nn.Module):
    def __init__(self, use_classification_layer=True, use_BG=False, num_classes=10, final_layer_bias=True):
        super(LeNet_plus_plus_5, self).__init__()
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
        self.fc1 = nn.Linear(
            in_features=self.conv3_2.out_channels * 3 * 3, out_features=5, bias=True
        )
        if use_classification_layer:
            if use_BG:
                self.fc2 = nn.Linear(
                    in_features=5, out_features=num_classes + 1, bias=final_layer_bias
                )
            else:
                self.fc2 = nn.Linear(in_features=5, out_features=num_classes, bias=final_layer_bias)
        self.use_classification_layer = use_classification_layer
        self.prelu_act1 = nn.PReLU()
        self.prelu_act2 = nn.PReLU()
        self.prelu_act3 = nn.PReLU()

    def forward(self, x):
        x = self.prelu_act1(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        x = self.prelu_act2(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        x = self.prelu_act3(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)

        y = self.fc1(x) # Features
        if self.use_classification_layer:
            x = self.fc2(y) # Logits
            return x, y
        return y
    
    def deep_feature_forward(self, y):
        return self.fc2(y)


class LeNet_plus_plus_10(nn.Module):
    def __init__(self, use_classification_layer=True, use_BG=False, num_classes=10, final_layer_bias=True):
        super(LeNet_plus_plus_10, self).__init__()
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
        self.fc1 = nn.Linear(
            in_features=self.conv3_2.out_channels * 3 * 3, out_features=10, bias=True
        )
        if use_classification_layer:
            if use_BG:
                self.fc2 = nn.Linear(
                    in_features=10, out_features=num_classes + 1, bias=final_layer_bias
                )
            else:
                self.fc2 = nn.Linear(in_features=10, out_features=num_classes, bias=final_layer_bias)
        self.use_classification_layer = use_classification_layer
        self.prelu_act1 = nn.PReLU()
        self.prelu_act2 = nn.PReLU()
        self.prelu_act3 = nn.PReLU()

    def forward(self, x):
        x = self.prelu_act1(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        x = self.prelu_act2(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        x = self.prelu_act3(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)

        y = self.fc1(x) # Features
        if self.use_classification_layer:
            x = self.fc2(y) # Logits
            return x, y
        return y
    
    def deep_feature_forward(self, y):
        return self.fc2(y)

