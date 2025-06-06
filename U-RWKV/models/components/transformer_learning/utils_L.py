import torch

import pandas as pd
from sklearn.model_selection import train_test_split
from dataset_L import MNISTTrainDataset, MNISTValDataset, MNISTSubmissionDataset

import numpy as np
from torch.utils.data import DataLoader, Dataset
import argparse


def get_loaders(train_df_dir, test_df_dir, submission_df_dir, batch_size):
    train_df = pd.read_csv(train_df_dir)
    test_df = pd.read_csv(test_df_dir)
    submission_df = pd.read_csv(submission_df_dir)

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    train_dataset = MNISTTrainDataset(train_df.iloc[:, 1:].values.astype(np.uint8), train_df.iloc[:, 0].values,
                                      train_df.index.values)
    val_dataset = MNISTValDataset(val_df.iloc[:, 1:].values.astype(np.uint8), val_df.iloc[:, 0].values,
                                  val_df.index.values)
    test_dataset = MNISTSubmissionDataset(test_df.iloc[:, 1:].values.astype(np.uint8), test_df.index.values)

    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    
    return train_dataloader, val_dataloader, test_dataloader




def get_arg_parser():
    parser = argparse.ArgumentParser(description="Vision Transformer (ViT) Training Script")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_df_dir', type=str, default="./dataset/train.csv", help='Path to train data')
    parser.add_argument('--test_df_dir', type=str, default="./dataset/test.csv", help='Path to test data')
    parser.add_argument('--submission_df_dir', type=str, default="./dataset/sample_submission.csv", help='Path to submission data')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--img_size', type=int, default=28, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.001, help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--activation', type=str, default="gelu", help='Activation function')
    parser.add_argument('--num_encoders', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd"], help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--datasetname', type=str, default="MNIST", help='Name of the dataset')
    parser.add_argument('--modelname', type=str, default="vit", help='Name of the model')
    return parser

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_accuracy(SR, GT):
    SR = torch.argmax(SR, dim=1)
    GT = GT.view(-1)
    correct = (SR == GT).sum().item()
    total = GT.size(0)
    acc = correct / total
    return acc