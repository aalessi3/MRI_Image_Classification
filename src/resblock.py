'''This is the basic building block for the resent network
5 of these blocks are stacked atop one another to form Resnet18'''

import torch.nn as nn

'''Whenever transitioning between layers we change the number of channels
from x to 2x (64 to 128 for example), in order to do this we need to drop the 
size of the image by 1/2. This is achived by stride length of two for our DownSampling
'''
class ResBlock(nn.Module):

    def __init__(self, in_chan, out_chan, downsample):
        super(ResBlock, self).__init__
        
        if downsample:
            stride = 2
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, 3, 2, 1),
                nn.BatchNorm2d(out_chan),
            )
        else:
            stride = 1
            self.shortcut = nn.Sequential()

        self.main = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, 3, stride, 1),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chan, out_chan, 3, 1, 1),
                nn.BatchNorm2d(out_chan),
            )
        self.relu = nn.ReLU
    
    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))

