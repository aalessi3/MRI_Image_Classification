import torch
import torch.utils.data
import torchvision
from torchmetrics import Accuracy
from torchvision import transforms
import numpy as np

dataroot = '../dataset/OriginalDataset'

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

image_size = 224
mean = 0
std = 1

accuracy = Accuracy(num_classes=4).to(device)

model = torch.load('../models/ResNet_E[35].pth').to(device)

dataset = torchvision.datasets.ImageFolder(root = dataroot, 
                                                transform=transforms.Compose([
                                                    transforms.Resize(image_size), 
                                                    transforms.CenterCrop(image_size), 
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize((mean, mean, mean), (std, std, std))
                                                ]))

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=50, shuffle=True)
acc = []
for i, (X,y) in enumerate(dataloader,0):
    with torch.no_grad():
        output = model(X.to(device))
    acc .append(accuracy(output, y.to(device)).item())
finalAcc = np.array(acc)
print(finalAcc.mean())
