import torch.nn as nn
import torch

class ImageSelfAttention_4(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0) #[N, 1, w, h]
        self.key = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.soft_max= nn.Softmax(dim=-1) #Soft max over rows
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=input_channels, kernel_size=1, stride=1, padding=0) #May not use or may modifty
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x):
        self.n = torch.tensor(x.size(2) * x.size(3)) #Total number of features
        # q = self.flatten(self.query(x).squeeze()) #[N, w, h]
        # k = self.flatten(self.key(x).squeeze())
        # v = self.flatten(self.value(x).squeeze())
        q = self.query(x).squeeze()
        k = self.query(x).squeeze()
        v = self.query(x).squeeze()
        # print(q.size())
        # print(k.size())
        # attend = torch.matmul(q.transpose(0,1), k) / torch.sqrt(self.n) #[]
        attend = torch.mul(q, k) / torch.sqrt(self.n)
        # print(attend.size())
        attend = self.flatten(attend)
        # print(attend.size())
        attend = self.soft_max(attend)
        attend = attend.view(x.size(0), x.size(2), x.size(3))
        # print(attend.size())
        content = torch.mul(attend, v).unsqueeze(1)
        # print(content.size())
        return self.final_conv(content) + x, attend


class ImageSelfAttention_3(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0) #[N, 1, w, h]
        self.key = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.soft_max= nn.Softmax(dim=-1) #Soft max over rows
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=input_channels, kernel_size=1, stride=1, padding=0) #May not use or may modifty
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x):
        self.n = torch.tensor(x.size(2) * x.size(3)) #Total number of features
        # q = self.flatten(self.query(x).squeeze()) #[N, w, h]
        # k = self.flatten(self.key(x).squeeze())
        # v = self.flatten(self.value(x).squeeze())
        q = self.query(x).squeeze()
        k = self.query(x).squeeze()
        v = self.query(x).squeeze()
        # print(q.size())
        # print(k.size())
        # attend = torch.matmul(q.transpose(0,1), k) / torch.sqrt(self.n) #[]
        attend = torch.mul(q, k) / torch.sqrt(self.n)
        # print(attend.size())
        attend = self.flatten(attend)
        # print(attend.size())
        attend = self.soft_max(attend)
        attend = attend.view(x.size(0), x.size(2), x.size(3))
        # print(attend.size())
        content = torch.mul(attend, v).unsqueeze(1)
        # print(content.size())
        return self.final_conv(content) + x, attend

class ImageSelfAttention(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0) #[N, 1, w, h]
        self.key = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.soft_max= nn.Softmax(dim=-1) #Soft max over rows
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=input_channels, kernel_size=1, stride=1, padding=0) #May not use or may modifty
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x):
        self.n = torch.tensor(x.size(2) * x.size(3)) #Total number of features
        q = self.query(x).squeeze() #[N, w, h]
        k = self.key(x).squeeze()
        v = self.value(x).squeeze()
        attend = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(self.n) #[]
        attend = self.flatten(attend)
        attend = self.soft_max(attend)
        attend = attend.view(x.size(0), x.size(2), x.size(3))
        return self.final_conv(torch.bmm(attend, v).unsqueeze(1)) + x, attend
    

class ImageSelfAttention_2(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0) #[N, 1, w, h]
        self.key = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.soft_max= nn.Softmax(dim=-1) #Soft max over rows
        self.final_conv = nn.Conv2d(in_channels=1, out_channels=input_channels, kernel_size=1, stride=1, padding=0) #May not use or may modifty
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x):
        self.n = torch.tensor(x.size(2) * x.size(3)) #Total number of features
        q = self.query(x).squeeze() #[N, w, h]
        k = self.key(x).squeeze()
        v = self.value(x).squeeze()
        attend = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(self.n) #[]
        attend = self.flatten(attend)
        attend = self.soft_max(attend)
        attend = attend.view(x.size(0), x.size(2), x.size(3))
        return self.final_conv(torch.bmm(attend, v).unsqueeze(1)) + x
    

class MCImageSelfAttention(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, padding=0) #[N, 1, w, h]
        self.key = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, padding=0)
        self.soft_max = nn.Softmax(dim=-1) #Soft max over columns
        self.upsample= nn.Upsample(scale_factor=4)

    def forward(self, x):
        self.n = torch.tensor(x.size(2) * x.size(3)) #Total number of features
        q = self.query(x)#[N, C, w, h]
        k = self.key(x)
        v = self.value(x)
        attend = torch.matmul(q, k.transpose(2, 3)) / torch.sqrt(self.n) #[]
        attend = self.soft_max(attend)
        return torch.matmul(attend, v) + x
        
class basic_CNN(nn.Module):
    def __init__(self, ):
        super(basic_CNN, self).__init__()
        self.attend = ImageSelfAttention_3(3)
        self.main = nn.Sequential(
            
            nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1), #[B x 3 x 224 x 224] ---> [B x 6 x 224 X 224]
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1), # [B x 6 x 224 x 224]  --> [B x 6 x 112 x 112]
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # ImageSelfAttention_2(6),
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1), #output : [B x 12 x 56 x 56]
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            # ImageSelfAttention(12),

            # ImageSelfAttention(12),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            # nn.AvgPool2d(kernel_size=3, stride=2, padding=1), #output : [B x 24 x 28 x 28]
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            # MCImageSelfAttention(24),
            # ImageSelfAttention(24),
            nn.Flatten(1), #output [B x 24*28*28]
            nn.Linear(24*28*28, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 4), 
            nn.Softmax(dim = 1)
        )
    
    def forward(self, x):
        x, attend = self.attend(x)
        return self.main(x), attend


    