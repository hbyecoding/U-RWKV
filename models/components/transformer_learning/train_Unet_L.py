import torch
import torch.nn as nn
from torch import optim
import timeit
from tqdm import tqdm
from utils_L import get_loaders,get_arg_parser, count_params, AverageMeter, get_accuracy
from model_L import Vit
import argparse
import wandb
from loguru import logger as log
import os
import datetime

# Hyper Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Parse command line arguments
parser = get_arg_parser()
args = parser.parse_args()

"""
args.in_channels
"""



# Model Parameters
IMG_SIZE = args.img_size
IN_CHANNELS = args.in_channels
PATCH_SIZE = args.patch_size
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
DROPOUT = args.dropout

NUM_HEADS = args.num_heads
ACTIVATION = args.activation
NUM_ENCODERS = args.num_encoders
NUM_CLASSES = args.num_classes

# LEARNING_RATE = args.learning_rate
# ADAM_WEIGHT_DECAY = args.adam_weight_decay
# ADAM_BETAS = args.adam_betas
# optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)

# Data Loaders
train_dataloader, val_dataloader, test_dataloader = get_loaders(args.train_df_dir, args.test_df_dir, args.submission_df_dir, args.batch_size)

# Model
model = Vit(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT,
            NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES).to(device)



criterion = nn.CrossEntropyLoss()

# Optimizer
if args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

# Logging setup
today_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_dir = f"{today_date}_log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{args.datasetname}_{args.modelname}.log")

log.add(log_file, format="{time} - {message}")
# Calculate model parameters
log.info(f"Model Parameters: {count_params(model)}")
# WandB setup
wandb.init(project="vit_training", config=args)

start = timeit.default_timer()

# Training loop
max_iterations = args.epochs * len(train_dataloader)
for epoch in tqdm(range(args.epochs), position=0, leave=True):
    model.train()
    train_loss_meter = AverageMeter()
    train_acc_meter = AverageMeter()
    for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_label["image"].float().to(device)
        label = img_label["label"].type(torch.long).to(device)
        y_pred = model(img)

        loss = criterion(y_pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_meter.update(loss.item(), img.size(0))
        train_acc = get_accuracy(y_pred, label)
        train_acc_meter.update(train_acc, img.size(0))

        # Update learning rate
        iter_num = epoch * len(train_dataloader) + idx
        # if args.optimizer == "sgd":
        #     lr_ = args.learning_rate * (1.0 - iter_num / max_iterations) ** 0.9
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_

    train_loss = train_loss_meter.avg
    train_accuracy = train_acc_meter.avg

    model.eval()
    val_loss_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    with torch.no_grad():
        for img_label in tqdm(val_dataloader, position=0, leave=True):
            img = img_label["image"].float().to(device)
            label = img_label["label"].type(torch.long).to(device)
            y_pred = model(img)

            loss = criterion(y_pred, label)
            val_loss_meter.update(loss.item(), img.size(0))
            val_acc = get_accuracy(y_pred, label)
            val_acc_meter.update(val_acc, img.size(0))

    val_loss = val_loss_meter.avg
    val_accuracy = val_acc_meter.avg

    log.info("-" * 30)
    log.info(f"Train Loss Epoch {epoch+1} : {train_loss:.4f}")
    log.info(f"Val Loss Epoch {epoch+1} : {val_loss:.4f}")
    log.info(f"Train Accuracy EPOCH {epoch + 1}: {train_accuracy:.4f}")
    log.info(f"Val Accuracy EPOCH {epoch + 1}: {val_accuracy:.4f}")
    log.info("-" * 30)

    # Log to WandB
    wandb.log({
        "Train Loss": train_loss,
        "Val Loss": val_loss,
        "Train Accuracy": train_accuracy,
        "Val Accuracy": val_accuracy,
        "Learning Rate": lr_
    })

stop = timeit.default_timer()
log.info(f"Training Time:{stop - start:.2f}s")

wandb.finish()