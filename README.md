# MRI_Image_Classification - Using Deep Neural Networks to Detect Dementia
aalessi3, rsiege15, ksuzuki6

This code base enables the contruction, training and evaluation of multiple deep neural network architectures. The task of these networks is to classify MRI brain scans into classes representing varying severities of dementia. The effects of residual connections as well as scaled dot product attention is investigated .The data used for training and evaluation is taken from Kaggle cited below:

Uraninjo. Augmented Alzheimer MRI Dataset. Retrieved 30 October, 2022 from https://www.kaggle.com/datasets/uraninjo/augmentedalzheimer-
mri-dataset.


# Running the Code:
1. After cloning the repository navigate the to the src directory
2. In order to train the models you must first download and save the kaggle data set
3. main_std_split.py is used to construct, train and save all network architectures. Replace dataroot relative path to the uncompressed DS.
4. Uncomment lines 190-193 depending upon which network you would like to train and run script
5. Models will be saved in a generated models folder and tensorboard .event files are saved in a generated tensorboard folder during training
6. Inference.py can be used to perform inference using any given model.pth file and dataroot. 

Alex - Test Edit


# Environment:
Created using python 3.10.6, all dependencies included in requirments.txt
