import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import wandb
from datetime import datetime
from pathlib import Path

from src.utils.losses import DiceLoss, BCEDiceLoss, IOUDiceLoss, BCEIOUDiceLoss
from src.utils.metrics import iou_score
from src.utils.util import AverageMeter
from src.dataloader.dataset import get_dataloader

def seed_everything(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description='U-RWKV Training')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='urwkv',
                      choices=['urwkv', 'urwkv_attention', 'urwkv_vit', 'urwkv_fusion',
                              'unet', 'unext', 'unet3plus',
                              'dwconv_fusion'],
                      help='Model architecture')
    parser.add_argument('--dims', type=str, default='24_48_96_192_384',
                      help='Model dimensions for each layer')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='busi',
                      choices=['busi', 'isic18', 'isic19', 'polyp'],
                      help='Dataset name')
    parser.add_argument('--img-size', type=int, default=256,
                      help='Input image size')
    parser.add_argument('--num-classes', type=int, default=1,
                      help='Number of classes')
    
    # Loss parameters
    parser.add_argument('--loss', type=str, default='dice',
                      choices=['dice', 'bce_dice', 'iou_dice', 'bce_iou_dice'],
                      help='Loss function')
    
    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=10,
                      help='How many batches to wait before logging')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='How many epochs to wait before saving')
    parser.add_argument('--experiment', type=str, default='default',
                      help='Experiment name for logging')
    
    return parser.parse_args()

def get_model(args):
    """Get model based on arguments"""
    dims = parse_list_arg(args.dims)
    depths = parse_list_arg(args.depths)
    kernels = parse_list_arg(args.kernels)
    
    if args.model == 'urwkv':
        from models.urwkv.base import URWKV
        return URWKV(dims=dims, num_classes=args.num_classes)
    elif args.model == 'urwkv_attention':
        from models.urwkv.attention import URWKV_Attention
        return URWKV_Attention(dims=dims, num_classes=args.num_classes)
    elif args.model == 'urwkv_vit':
        from models.urwkv.vit import URWKV_ViT
        return URWKV_ViT(dims=dims, num_classes=args.num_classes)
    elif args.model == 'urwkv_fusion':
        from models.urwkv.fusion import URWKV_Fusion
        return URWKV_Fusion(dims=dims, num_classes=args.num_classes)
    elif args.model == 'unet':
        from models.variants.unet import UNet
        return UNet(output_ch=args.num_classes)
    elif args.model == 'unext':
        from models.variants.unext import UNext
        return UNext(num_classes=args.num_classes)
    elif args.model == 'unet3plus':
        from models.variants.unet3plus import UNet3plus
        return UNet3plus(n_classes=args.num_classes)
    elif args.model == 'dwconv_fusion':
        from models.variants.dwconv import DWConvFusion
        return DWConvFusion(dims=dims, depths=depths, kernels=kernels, num_classes=args.num_classes)
    else:
        raise ValueError(f"Model {args.model} not supported")

def get_loss_fn(loss_type):
    """Get loss function based on type"""
    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'bce_dice':
        return BCEDiceLoss()
    elif loss_type == 'iou_dice':
        return IOUDiceLoss()
    elif loss_type == 'bce_iou_dice':
        return BCEIOUDiceLoss()
    else:
        raise ValueError(f"Loss type {loss_type} not supported")

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """Train for one epoch"""
    model.train()
    avg_meters = {'loss': AverageMeter(),
                 'iou': AverageMeter(),
                 'dice': AverageMeter()}
    
    for batch_idx, batch in enumerate(train_loader):
        images, targets = batch['image'].to(device), batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Calculate metrics
        iou, dice, _, _, _, _, _ = iou_score(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        # Update meters
        avg_meters['loss'].update(loss.item(), images.size(0))
        avg_meters['iou'].update(iou, images.size(0))
        avg_meters['dice'].update(dice, images.size(0))
        
        # Log progress
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {avg_meters["loss"].avg:.6f}\t'
                  f'IoU: {avg_meters["iou"].avg:.6f}\t'
                  f'Dice: {avg_meters["dice"].avg:.6f}')
    
    return avg_meters

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    avg_meters = {'val_loss': AverageMeter(),
                 'val_iou': AverageMeter(),
                 'val_dice': AverageMeter(),
                 'val_se': AverageMeter(),
                 'val_pc': AverageMeter(),
                 'val_f1': AverageMeter(),
                 'val_acc': AverageMeter()}
    
    with torch.no_grad():
        for batch in val_loader:
            images, targets = batch['image'].to(device), batch['label'].to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Calculate metrics
            iou, dice, se, pc, f1, _, acc = iou_score(outputs, targets)
            
            # Update meters
            avg_meters['val_loss'].update(loss.item(), images.size(0))
            avg_meters['val_iou'].update(iou, images.size(0))
            avg_meters['val_dice'].update(dice, images.size(0))
            avg_meters['val_se'].update(se, images.size(0))
            avg_meters['val_pc'].update(pc, images.size(0))
            avg_meters['val_f1'].update(f1, images.size(0))
            avg_meters['val_acc'].update(acc, images.size(0))
    
    return avg_meters

def main():
    args = get_args()
    seed_everything(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb
    wandb.init(
        project="U-RWKV",
        name=f"{args.experiment}_{args.model}_{args.dataset}",
        config=args.__dict__
    )
    
    # Create model
    model = get_model(args).to(device)
    
    # Get data loaders
    train_loader, val_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # Setup training
    criterion = get_loss_fn(args.loss).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_iou = 0
    for epoch in range(1, args.epochs + 1):
        # Train
        train_meters = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        
        # Validate
        val_meters = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'lr': scheduler.get_last_lr()[0],
            'train_loss': train_meters['loss'].avg,
            'train_iou': train_meters['iou'].avg,
            'train_dice': train_meters['dice'].avg,
            'val_loss': val_meters['val_loss'].avg,
            'val_iou': val_meters['val_iou'].avg,
            'val_dice': val_meters['val_dice'].avg,
            'val_se': val_meters['val_se'].avg,
            'val_pc': val_meters['val_pc'].avg,
            'val_f1': val_meters['val_f1'].avg,
            'val_acc': val_meters['val_acc'].avg
        }
        wandb.log(metrics)
        
        # Save best model
        if val_meters['val_iou'].avg > best_iou:
            best_iou = val_meters['val_iou'].avg
            checkpoint_dir = Path(f'checkpoints/{args.experiment}')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = checkpoint_dir / f'best_model_{args.model}_{args.dataset}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': args.__dict__
            }, model_path)
            
            print(f'=> Saved best model with IoU: {best_iou:.4f}')
        
        # Print epoch summary
        print(f'Epoch {epoch}/{args.epochs}:')
        print(f'Train - Loss: {metrics["train_loss"]:.4f}, IoU: {metrics["train_iou"]:.4f}, Dice: {metrics["train_dice"]:.4f}')
        print(f'Val - Loss: {metrics["val_loss"]:.4f}, IoU: {metrics["val_iou"]:.4f}, Dice: {metrics["val_dice"]:.4f}')
        print(f'Val - SE: {metrics["val_se"]:.4f}, PC: {metrics["val_pc"]:.4f}, F1: {metrics["val_f1"]:.4f}, ACC: {metrics["val_acc"]:.4f}')
    
    wandb.finish()

if __name__ == '__main__':
    main() 