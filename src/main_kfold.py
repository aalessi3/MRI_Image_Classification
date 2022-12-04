import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from ResNet18 import ResNet
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from sklearn.model_selection import KFold

n_splits = 10
kfold = KFold(n_splits=n_splits, shuffle=True)
writer = SummaryWriter(log_dir='../tensorboard')

#Use GPU if your PC is configured to do so (i.e you have a Nvidia capable GPU, cuda toolkit and cudnn installed and pytorch with cuda installed)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


#TODO This initizes weights in a simplier fashion than is 
#done in the paper, not sure if this will cause performence issues
def weight_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

#TODO finish training loop
def train(model, num_epoch, dataloader, dataset, batchSize):
    criterion = torch.nn.CrossEntropyLoss()
    #SGD with the following params is specified in the paper
    
    accuracy = Accuracy(num_classes=4).to(device)
    #Book keeping 
    runningLoss = 0
    for fold, (train_id, test_id) in enumerate(kfold.split(dataset)):
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_id)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_id)

        trainLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchSize, sampler= train_subsampler)
        testLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchSize, sampler= test_subsampler)

        optomizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9 )

        for epoch in range(num_epoch):
        #for i, (X,y) in enumerate(dataloader, 0):
            runningLoss = 0
            for i, data in enumerate(trainLoader, 0):
                input, target = data
                #print(input.shape)
            #Zero gradients after reach batch 
                optomizer.zero_grad()

                output = model(input.to(device))
                loss = criterion(output, target.to(device))
                loss.backward()
                optomizer.step()
                runningLoss += loss
            
                
            print(f"[{epoch +1}\{num_epoch}]:\t Loss: {runningLoss}")
        #writer.add_scalar("Loss_Train", runningLoss/len(dataloader), epoch)
        
        torch.save(model, f'../models/ResNet_E_Fold[{fold+1}].pth')
        acc = []
        for i, data in enumerate(testLoader, 0):
            input, target = data
            output = model(input.to(device))

            acc.append(accuracy(output, target.to(device)).cpu())

        accNP = np.array(acc)
        print(f'Folds : {fold} \t Accuracy {accNP.mean()}')




        
        



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
    #I choose batch size used in ResNet paper
    batch_size = 64

    #Nummber of threads used by dataloader object
    workers = 2

    #If true we will print one batch of images from dataloader
    printBatch = False

    #number of epochs for training
    num_epoch = 1

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

    '''Use this to visualize a batch of input images'''
    if printBatch:
        batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.imshow(np.transpose(torchvision.utils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

    '''create model and initialize weights'''
    model = ResNet(n_channels=n_channels, n_classes=n_classes)
    model.apply(weight_init).to(device)


    train(model = model, num_epoch = num_epoch, dataloader= dataloader, dataset=dataset, batchSize=batch_size)
    #write all pending data to writer and close it, 
    writer.flush()
    writer.close()




if __name__ == '__main__':
    main()