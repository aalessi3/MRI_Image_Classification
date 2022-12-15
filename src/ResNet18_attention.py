import torch.nn as nn
from resblock import ResBlock
# from attendBlock import imageAttentionBlock
from basic_CNN import ImageSelfAttention_2

class ResNet_A(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResNet_A, self).__init__()
        self.layer_0 = nn.Sequential(
            ImageSelfAttention_2(input_channels=n_channels),
            nn.Conv2d(in_channels=n_channels, out_channels=64,kernel_size= 7, stride= 2, padding= 3),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        self.layer_1 = nn.Sequential(
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 64, downsample=False),
        )

        # self.attend1 = imageAttentionBlock(n_head=2, n_layers=64, img_size=56)

        self.layer_2 = nn.Sequential(
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False),
        )

        # self.attend2 = imageAttentionBlock(n_head=2, n_layers=128, img_size=28)

        self.layer_3 = nn.Sequential(
            ResBlock(128, 256, downsample=True),
            ResBlock(256, 256, downsample=False),
        )

        # self.attend3 = imageAttentionBlock(n_head=2, n_layers= 256, img_size=14)

        self.layer_4 = nn.Sequential(
            ResBlock(256, 512, downsample=True), 
            ResBlock(512, 512, downsample=False)
        )

        # self.attend4 = imageAttentionBlock(n_head=2, n_layers=512, img_size=7)

        self.layer_5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, n_classes),
            nn.Softmax()
            
        )

    
    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        # x = self.attend1(x)
        x = self.layer_2(x)
        # x = self.attend2(x)
        x = self.layer_3(x)
        # x = self.attend3(x)
        x = self.layer_4(x)
        # x = self.attend4(x)
        return  self.layer_5(x)

