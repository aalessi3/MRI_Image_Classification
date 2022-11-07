import torch.nn as nn
from resblock import ResBlock

class ResNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNet, self).__init__()

        self.layer_0 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64,kernel_size= 7, stride= 2, padding= 3),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        self.layer_1 = nn.Sequential(
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False),
        )
        self.layer_2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False),
        )
        self.layer_3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False),
        )
        self.layer_4 = nn.Sequential(
            ResBlock(256, 512, downsample=True), 
            ResBlock(512, 512, downsample=False)
        )
        self.layer_5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=n_classes)
        )

    
    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return self.layer_5(x)