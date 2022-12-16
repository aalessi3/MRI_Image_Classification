import torch
import torch.utils.data
import torchvision
from torchmetrics import Accuracy
from torchvision import transforms
import numpy as np

'''
This script is used to evaluate the accuracy of a given model on a dataset. Simply load the desired model and dataset by providing the appropriate relative paths
'''

dataroot = '../dataset/AugmentedAlzheimerDataset'

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

image_size = 224
mean = 0
std = 1

accuracy = Accuracy(num_classes=4).to(device)

model = torch.load('../models_7_12_22/ResNet_E[94].pth').to(device)

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
