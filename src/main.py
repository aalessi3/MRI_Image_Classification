import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from ResNet18 import ResNet

#TODO write function to initialize network weights in the same way ResNet paper does
def weight_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv' != -1):
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def train(model, num_epoch, lr, dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    #SGD with the following params is specified in the paper
    optomizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9 )


def main():

    #Prints device being used (CPU or GPU) just so we know
    print(f'Using {torch.cuda.get_device_name()}')

    #Use GPU if your PC is configured to do so (i.e you have a Nvidia capable GPU, cuda toolkit and cudnn installed and pytorch with cuda installed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #Directory leading to image folders from cwd of script
    dataroot = '../dataset/AugmentedAlzheimerDataset'

    #Image size used in ResNet paper
    image_size = 224

    #Number of classes
    n_classes = 4

    #Number of channels in img
    n_channels = 3

    #Mean and std to normalize images to
    mean = 0
    std = 1

    #Batch size for training, too large and you may overload your GPU memory. 
    #I choose batch size used in ResNet paper
    batch_size = 256

    #Nummber of threads used by dataloader object
    workers = 2

    #If true we will print one batch of images from dataloader
    printBatch = True

    #number of epochs for training
    num_epoch = 5

    '''This is a dataset object, it will access all the photos in the subdirectories of root
    and when those photos are loaded it will perform the transformations sepcified by transform. 
    Each image will have a corresponding label dictated by the name of the folder it was found in'''
    dataset = torchvision.datasets.ImageFolder(root = dataroot, 
                                                transform=transforms.Compose([
                                                    transforms.Resize(image_size), 
                                                    transforms.CenterCrop(image_size), 
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize((mean, mean, mean), (std, std, std))
                                                ]))

    #Print some info about the dataset
    print(f'Image labels and label index : {dataset.class_to_idx}')
    print(f'Dataset size: {len(dataset)}')
    classes = dataset.classes

    '''This is a dataloader, it wraps an iterator around our dataset object. It will return photos in batches of batch_size along with associated labels.
    It can do so using multiple threads defined by num_workers'''
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size = batch_size, 
                                                shuffle = True, num_workers=workers)


    if printBatch:
        batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

    model = ResNet(n_channels=n_channels, n_classes=n_classes)
    model.apply(weight_init)

    train(model = model, num_epoch = num_epoch, dataloader= dataloader)




if __name__ == '__main__':
    main()