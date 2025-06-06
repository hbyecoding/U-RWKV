# -*- encoding: utf-8 -*-
'''
@File   :  benchmark_params.py
@Time   :  2024/12/01 21:33:48
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


from torchvision.models import resnet50

def count_params(model):
    out = sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == "__main__":
    from loguru import logger as log
    model = resnet50(pretrained=False)
    
    print(count_params(model))

    