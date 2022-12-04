'''This generates a test and train dataset folders'''

import os


dataRoot = "../dataset/AugmentedAlzheimerDataset"
validationRoot = "../validationDataset"


for folder in os.listdir(dataRoot):
    os.mkdirs(os.path.join(validationRoot, folder))
    classPath = os.path.join(dataRoot, folder)
    i = 0
    for img in os.listdir(classPath):
        
