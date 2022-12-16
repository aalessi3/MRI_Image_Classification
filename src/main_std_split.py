import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from ResNet18 import ResNet
from ResNet18_attention import ResNet_A
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, F1Score
from basic_CNN import basic_CNN
from CNN_18 import CNN_18
import os

'''
This script is used to initalize, train and test a given model. Model metrics are saved in tensorboard files and the best model .pth file at any given time is saved in ../models. 
'''



#Used to save tensorboard figures
writer = SummaryWriter(log_dir='../tensorboard')


#Use GPU if your PC is configured to do so (i.e you have a Nvidia capable GPU, cuda toolkit and cudnn installed and pytorch with cuda installed)
device = "cuda:0" if torch.cuda.is_available() else "cpu"



#This initizes weights in a simplier fashion than is 
#done in the paper
def weight_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def train(model, num_epoch, train_dataloader, val_dataloader, test_dataloader):
    print("Begining Training")
    criterion = torch.nn.CrossEntropyLoss()
    optomizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=0.9 )


    f1_score = F1Score(num_classes=4).to(device)
    accuracy = Accuracy(um_classes=4).to(device)

    #Book keeping 
    runningLoss = 0
    val_loss = 0
    running_f1 = 0
    acc = []

    best_loss = np.inf

    for epoch in range(num_epoch):
        for i, (X,y) in enumerate(train_dataloader, 0):
            #Zero gradients after each batch 
            model.train()
            optomizer.zero_grad()
            #Uncomment this line when attention model which returns attenton weights is used
            # output, _ = model(X.to(device))
            output= model(X.to(device))
            loss = criterion(output, y.to(device))
            loss.backward()
            optomizer.step()
            #Type casting to float avoids accumulating grad histories in runningLoss
            runningLoss += float(loss)
        
        for i, (X,y) in enumerate(val_dataloader, 0):
            model.eval()
            #Uncomment this line when attention model which returns attenton weights is used
            # output, _ = model(X.to(device))
            output= model(X.to(device))
            loss = criterion(output, y.to(device))
            val_loss += float(loss)
            running_f1 += f1_score(output, y.to(device))
            acc.append(accuracy(output, y.to(device)).cpu())
            
        
        f1 = running_f1/len(val_dataloader)
        avg_acc = np.array(acc).mean()
                           
        print(f"[{epoch +1}\{num_epoch}]:\t Train_Loss: {runningLoss / len(train_dataloader)} Val_Loss: {val_loss/len(val_dataloader)} F1: {f1} Accuracy: {avg_acc}")
        writer.add_scalar("Loss_Train", runningLoss/len(train_dataloader), epoch)
        writer.add_scalar("Loss_Val", val_loss/len(val_dataloader), epoch)
        writer.add_scalar("F1_Score", f1, epoch)
        writer.add_scalar("Accuracy", avg_acc, epoch)
        # writer.add_scalar("Accuracy", accuracy, epoch)
        if (not os.path.isdir("../models")):
            os.makedirs("../models")
        if(val_loss < best_loss):
            best_loss = val_loss
            torch.save(model, f'../models/ResNet_E[{epoch+1}].pth')
        runningLoss = 0
        val_loss = 0
        running_f1 = 0
        acc = []

    for i, (X,y) in enumerate(test_dataloader):
        model.eval()
        #Uncomment this line when attention model which returns attenton weights is used
        # output, _ = model(X.to(device))
        output= model(X.to(device))
        loss = criterion(output, y.to(device))
        val_loss += float(loss)
        acc.append(accuracy(output, y.to(device)).cpu())
    print(f"Final Test Accuracy {np.array(acc).mean()}")
        
        



def main():

    #Prints device being used (CPU or GPU) just so we know
    print(f'Using {torch.cuda.get_device_name()}')


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
    #I choose mac batchsize that does not overload mem
    batch_size = 64

    #Nummber of threads used by dataloader object
    workers = 2

    #If true we will print one batch of images from dataloader for visualization
    printBatch = False

    #number of epochs for training
    num_epoch = 100

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

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths= [0.7, 0.1, 0.2])

    '''This is a dataloader, it wraps an iterator around our dataset object. It will return photos in batches of batch_size along with associated labels.
    It can do so using multiple threads defined by num_workers'''
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_size, 
                                                shuffle = True, num_workers=workers)

    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size = batch_size, 
                                                shuffle = True, num_workers=workers)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = batch_size, 
                                                shuffle = True, num_workers=workers)

    '''Use this to visualize a batch of input images'''
    if printBatch:
        batch = next(iter(train_dataloader))
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

    '''create model and initialize weights. Uncomment a given line in order to train different models'''
    # model = ResNet(n_channels=n_channels, n_classes=n_classes)
    # model = ResNet_A(n_channels=n_channels, n_classes=n_classes)
    # model = basic_CNN()
    model = CNN_18(n_channels=n_channels, n_classes=n_classes).to(device)
    model.apply(weight_init).to(device)


    train(model = model, num_epoch = num_epoch,
             train_dataloader= train_dataloader,
             val_dataloader = val_dataloader,
             test_dataloader=test_dataloader,)
    #write all pending data to writer and close it
    writer.flush()
    writer.close()
 




if __name__ == '__main__':
    main()