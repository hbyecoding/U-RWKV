# -*- encoding: utf-8 -*-
'''
@File   :  how2trianmy_vit_.py
@Time   :  2024/12/07 17:37:13
@Author :  hbye
'''
######################################   外部调用   ######################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
#########################################################################################
######################################   内部调用   ######################################
#########################################################################################

# import  mnist

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms



import pandas as pd
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np


class MNISTTrainDataset(Dataset):
    def __init__(self, images, labels, indicies):
        self.images = images
        self.labels = labels
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indicies[idx]
        image = self.transform(image)

        return {"image":image, "label":label, "index":index}


class MNISTValDataset(Dataset):
    def __init__(self, images, labels, indicies):
        self.images = images
        self.labels = labels
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indicies[idx]
        image = self.transform(image)

        return {"image": image, "label": label, "index": index}


class MNISTSubmissionDataset(Dataset):
    def __init__(self, images, indicies):
        self.images = images
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        index = self.indicies[idx]
        image = self.transform(image)

        return {"image": image, "index": index}
    
    
    