import torch.nn as nn
import torch
from ImageSelfAttention import ImageSelfAttention_2

'''This is a basic 5 layer CNN used to evalaute the effects of self attention on a small, easy to train network. This was used in development because of a lack of large compute.

    ARGS:
        x : input image batch. expected to be 224x224 image

    Returns:
        attend : attention weight matrix
        result : tensor of len 4, rep probability of each class
    '''

class basic_CNN(nn.Module):
    def __init__(self, ):
        super(basic_CNN, self).__init__()

        self.attend = ImageSelfAttention_2(3)

        self.main = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1), #[B x 3 x 224 x 224] ---> [B x 6 x 112 X 112]
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
            nn.Linear(24*28*28, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 4), 
            nn.Softmax(dim = 1)
        )
    
    def forward(self, x):
        x, attend = self.attend(x) #Attention applied at first layer was found to be most effective
        return self.main(x), attend


    