import torch.nn as nn

class imageAttentionBlock(nn.Module):
    def __init__(self, n_head, n_layers, img_size):
        super(imageAttentionBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(n_layers, out_channels= 1,kernel_size= 3, stride= 1, padding= 1),
            nn.MultiheadAttention(num_heads=n_head, embed_dim= img_size*img_size),
            nn.Conv2d(1, n_layers, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.main(x)