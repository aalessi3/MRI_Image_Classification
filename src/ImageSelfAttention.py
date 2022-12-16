import torch.nn as nn
import torch

'''This is the self attention object used to apply scaled dot product attention to images

    ARGS:
        input_channels : Number of channels in input images
        x: input image batch with input_channels num of channels
    
    Returns:
        Conext : self attention matrix applied to the input images
        attend : Attention weight matrix
'''

class ImageSelfAttention(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        '''1x1 Conv is used to compress n channel input to single channel. Attention operation is too expensive to apply to all channels'''
        self.query = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.soft_max= nn.Softmax(dim=-1) 
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=input_channels, kernel_size=1, stride=1, padding=0) 
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x):
        self.n = torch.tensor(x.size(2) * x.size(3)) #Total number of features, used to scale attention weights
        q = self.query(x).squeeze() #[N, w, h]
        k = self.key(x).squeeze() #[N, w, h]
        v = self.value(x).squeeze() #[N, w, h]
        attend = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(self.n) #[N, w, w]
        attend = self.flatten(attend)
        attend = self.soft_max(attend)
        attend = attend.view(x.size(0), x.size(2), x.size(3)) #[N, w, h]
        return self.final_conv(torch.bmm(attend, v).unsqueeze(1)) + x, attend