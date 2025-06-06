# -*- encoding: utf-8 -*-
'''
@File   :  transformer.py
@Time   :  2024/12/07 16:33:44
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



if __name__ =="__main__":
    
    
    # 空格分词
    src_vocab = {"P":0, 'ich':1,'mochte' :3, 'bier':4, 'cola':5}
    src_vocab_size = len(src_vocab)
    # 句号也算一个token
    
    tgt_vocab = {"P":0, 'i':1,  'want' :3, 'a':4, 'beer':5, 'coke':6, 'S':7, '.':8}
    tgt_vocab_size = len(tgt_vocab)
    
    
    
    idx2word = {i : w for i, w in enumerate(tgt_vocab)}