import torch.nn as nn
from torchvision import models

class ResNet_18(nn.Module):
    def __init__(self, fc_layer_dim=1000, small_scale=True, use_classification_layer=True, use_BG=False, num_classes=10, final_layer_bias=True):
        super(ResNet_18, self).__init__()
        resnet_base = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        fc_in_features = resnet_base.fc.in_features
        resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.fc1 = resnet_base
        if use_classification_layer:
            if use_BG:
                self.fc2 = nn.Linear(
                    in_features=fc_layer_dim, out_features=num_classes + 1, bias=final_layer_bias
                )
            else:
                self.fc2 = nn.Linear(in_features=fc_layer_dim, out_features=num_classes, bias=final_layer_bias)
        self.use_classification_layer = use_classification_layer
        
    def forward(self, x):
        y = self.fc1(x) # Features
        if self.use_classification_layer:
            x = self.fc2(y) # Logits
            return x, y
        return y


class ResNet_50(nn.Module):
    def __init__(self, fc_layer_dim=1000, small_scale=True, use_classification_layer=True, use_BG=False, num_classes=10, final_layer_bias=True):
        super(ResNet_50, self).__init__()
        resnet_base = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        fc_in_features = resnet_base.fc.in_features
        resnet_base.fc = nn.Linear(in_features=fc_in_features, out_features=fc_layer_dim)

        self.fc1 = resnet_base
        if use_classification_layer:
            if use_BG:
                self.fc2 = nn.Linear(
                    in_features=fc_layer_dim, out_features=num_classes + 1, bias=final_layer_bias
                )
            else:
                self.fc2 = nn.Linear(in_features=fc_layer_dim, out_features=num_classes, bias=final_layer_bias)
        self.use_classification_layer = use_classification_layer
        
    def forward(self, x):
        y = self.fc1(x) # Features
        if self.use_classification_layer:
            x = self.fc2(y) # Logits
            return x, y
        return y

########################################################################
# Author: Vision And Security Technology (VAST) Lab in UCCS
# Date: 2024
# Availability: https://github.com/Vastlab/vast?tab=readme-ov-file
########################################################################

class LeNet_plus_plus(nn.Module):
    def __init__(self, use_classification_layer=True, small_scale=True, use_BG=False, num_classes=10, final_layer_bias=True):
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
        self.fc1 = nn.Linear(
            in_features=self.conv3_2.out_channels * 3 * 3, out_features=2, bias=True
        )
        if use_classification_layer:
            if use_BG:
                self.fc2 = nn.Linear(
                    in_features=2, out_features=num_classes + 1, bias=final_layer_bias
                )
            else:
                self.fc2 = nn.Linear(in_features=2, out_features=num_classes, bias=final_layer_bias)
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


# class LeNet(nn.Module):
#     def __init__(self, use_classification_layer=True, use_BG=False, num_classes=10, final_layer_bias=True):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=20, kernel_size=(5, 5), stride=1, padding=2
#         )
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
#         self.conv2 = nn.Conv2d(
#             in_channels=self.conv1.out_channels,
#             out_channels=50,
#             kernel_size=(5, 5),
#             stride=1,
#             padding=2,
#         )
#         self.fc1 = nn.Linear(
#             in_features=self.conv2.out_channels * 7 * 7, out_features=500, bias=True
#         )
#         if use_classification_layer:
#             if use_BG:
#                 self.fc2 = nn.Linear(
#                     in_features=500, out_features=num_classes + 1, bias=final_layer_bias
#                 )
#             else:
#                 self.fc2 = nn.Linear(in_features=500, out_features=num_classes, bias=final_layer_bias)
#         self.relu_act = nn.ReLU()
#         self.use_classification_layer = use_classification_layer
#         print(
#             f"{' Model Architecture '.center(90, '#')}\n{self}\n{' Model Architecture End '.center(90, '#')}"
#         )

#     def forward(self, x):
#         x = self.pool(self.relu_act(self.conv1(x)))
#         x = self.pool(self.relu_act(self.conv2(x)))
#         x = x.view(-1, self.conv2.out_channels * 7 * 7)
#         y = self.fc1(x)
#         if self.use_classification_layer:
#             x = self.fc2(y)
#             return x, y
#         return y
