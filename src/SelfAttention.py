"""
This is Self Attention class used for CNNs
Resource: https://medium.com/mlearning-ai/self-attention-in-convolutional-neural-networks-172d947afc00
Last update: 2022/11/16

Function: This class will allow the model to increase the receptivie field of the CNN withoutt adding computational cost associated with very large kernel sizes
input: CNN feature maps
output: Updated features maps with applied self-learning
"""
import pytorch

class SelfAttention(Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(tensor([0.]))

    def _conv(self,n_in,n_out):
        #https://rdrr.io/cran/fastai/man/ConvLayer.html
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, act_cls=None, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1) #batch matrix multiply
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

class SelfAttn():
    def __init__(self, n_channels):
        self.query, self.key, self.value = [self.conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(tensor([0.]))

    def self_atten(self, x):

''' alternative (module in pytorch)
https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
self_atention = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = self_attention(query, key, value);
'''


''' example calling this class
SelfAttention self_attn
self_attn.conv(input)
output = self_attn.forward(self_attn.conv)
'''
