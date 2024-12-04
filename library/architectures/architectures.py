import torch.nn as nn
from torchvision import models

class ResNet_50(nn.Module):
    ########################################################################
    # Reference: 
    # Vision And Security Technology (VAST) Lab in UCCS
    # https://github.com/Vastlab/vast?tab=readme-ov-file
    ########################################################################

    def __init__(self, feat_dim=-1, num_classes=10, final_layer_bias=False, is_verbose=True):

        if is_verbose:
            print("\n↓↓↓ Architecture setup ↓↓↓")
            print(f"{self.__class__.__name__} Architecture Loaded!")
            print(f"Set deep feature dimension to {1000 if feat_dim == -1 else feat_dim}")
            if final_layer_bias: print('Classifier has a bias term.')

        super(ResNet_50, self).__init__()

        # Feature Extractor (F)
        resnet_base = models.resnet50(weights=None)
        fc_in_features = resnet_base.fc.in_features
        resnet_base.fc = nn.Linear(in_features=fc_in_features, 
                                   out_features=1000 if feat_dim == -1 else feat_dim)
        self.fc1 = resnet_base

        # Classification Head (H)
        self.fc2 = nn.Linear(in_features=1000 if feat_dim == -1 else feat_dim, 
                            out_features=num_classes, bias=final_layer_bias)
        
    def forward(self, x):
        y = self.fc1(x) # Features
        x = self.fc2(y) # Logits
        return x, y

class LeNet_plus_plus(nn.Module):

    ########################################################################
    # Reference: 
    # Vision And Security Technology (VAST) Lab in UCCS
    # https://github.com/Vastlab/vast?tab=readme-ov-file
    ########################################################################

    def __init__(self, feat_dim=-1, num_classes=10, final_layer_bias=False, is_verbose=True):
        
        if is_verbose: 
            print("\n↓↓↓ Architecture setup ↓↓↓")
            print(f"{self.__class__.__name__} Architecture Loaded!")
            print(f"Set deep feature dimension to {2 if feat_dim == -1 else feat_dim}")
            if final_layer_bias: print('Classifier has a bias term.')

        super(LeNet_plus_plus, self).__init__()



        # Feature Extractor (F)
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
        self.prelu_act1 = nn.PReLU()
        self.prelu_act2 = nn.PReLU()
        self.prelu_act3 = nn.PReLU()
        self.fc1 = nn.Linear(in_features=self.conv3_2.out_channels * 3 * 3,
                             out_features=2 if feat_dim == -1 else feat_dim)



        # Classification Head (H)
        self.fc2 = nn.Linear(in_features=2 if feat_dim == -1 else feat_dim, 
                            out_features=num_classes, bias=final_layer_bias)

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

    ########################################################################
    # Reference: 
    # Vision And Security Technology (VAST) Lab in UCCS
    # https://github.com/Vastlab/vast?tab=readme-ov-file
    ########################################################################

    def __init__(self, use_BG=False, feat_dim=-1, num_classes=10, final_layer_bias=False, is_osovr=False, is_verbose=True):

        if is_verbose:
            print("\n↓↓↓ Architecture setup ↓↓↓")
            print(f"{self.__class__.__name__} Architecture Loaded!")
            print(f"Set deep feature dimension to {500 if feat_dim == -1 else feat_dim}")
            if final_layer_bias: print('Classifier has a bias term.')

        super(LeNet, self).__init__()


        # Feature Extractor (F)
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
        self.relu_act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(
            in_features=self.conv2.out_channels * 7 * 7, 
            out_features=500 if feat_dim == -1 else feat_dim, bias=True
        )

        # Classification Head (H)
        self.fc2 = nn.Linear(in_features=500 if feat_dim == -1 else feat_dim, 
                            out_features=num_classes, bias=final_layer_bias)
        
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
