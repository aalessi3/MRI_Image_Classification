import torch
import torchvision 
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np

#TODO write function to initialize network weights in the same way ResNet paper does
def weight_init():
    raise NotImplementedError

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
    nc = 4

    #Mean and std to normalize images to
    mean = 0
    std = 1

    #Batch size for training, too large and you may overload your GPU memory. 
    #I choose batch size used in ResNet paper
    batch_size = 256

    #Nummber of threads used by dataloader object
    workers = 2


    #If true we will print one batch of images from dataloader
    printBatch = False

    '''This is a dataset object, it will access all the photos in the subdirectories of root
    and when those photos are loaded it will perform the transformations sepcified by transform. 
    Each image will have a corresponding label dictated by the name of the folder it was found in'''
    dataset = torchvision.datasets.ImageFolder(root = dataroot, 
                                                transform=transforms.Compose([
                                                    transforms.Resize(image_size), 
                                                    transforms.CenterCrop(image_size), 
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize((mean), (std))
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



if __name__ == '__main__':
    main()