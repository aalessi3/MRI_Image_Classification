import torch.nn as nn
import torch

class basic_CNN(nn.Module):
    def __init__(self, ):
        super(basic_CNN, self).__init__()
        # self.main = nn.Sequential(
        #     nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1), #[B x 3 x 224 x 224] ---> [B x 6 x 224 X 224]
        #     nn.AvgPool2d(kernel_size=3, stride=2, padding=1), # [B x 6 x 224 x 224]  --> [B x 6 x 112 x 112]
        #     # nn.BatchNorm2d(6),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
        #     nn.AvgPool2d(kernel_size=3, stride=2, padding=1), #output : [B x 12 x 56 x 56]
        #     # nn.BatchNorm2d(6),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1),
        #     nn.AvgPool2d(kernel_size=3, stride=2, padding=1), #output : [B x 24 x 28 x 28]
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(1), #output [B x 24*28*28]
        #     nn.Linear(12*28*28, 1000),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1000, 4), 
        #     nn.Softmax(dim = 1)
        # )

        self.main = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1), #[B x 3 x 224 x 224] ---> [B x 6 x 224 X 224]
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1), # [B x 6 x 224 x 224]  --> [B x 6 x 112 x 112]
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1), #output : [B x 12 x 56 x 56]
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1), #output : [B x 24 x 28 x 28]
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Flatten(1), #output [B x 24*28*28]
            nn.Linear(24*28*28, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 4), 
            nn.Softmax(dim = 1)
        )
    
    def forward(self, x):
        return self.main(x)

    