# -*- encoding: utf-8 -*-
'''
@File   :  transformer.py
@Time   :  2024/12/07 16:41:14
@Author :  hbye
'''
######################################   外部调用   ######################################


import math

# data
import  torch.optim as optim
import  torch.utils.data as Data


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


# class MyDataset

class MyDataset(Data.Dataset):
    def __init__(self, enc_input, dec_inputs, dec_outputs):
        super().__init__()
        
        self.enc_input = enc_input
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs
        
    
    def __getitem__(self,index):
        pass


loader = Data.DataLoader(MyDataset(enc_input, dec_inputs, dec_outputs), 2, shuffle=True)

model = Transformer().cuda()