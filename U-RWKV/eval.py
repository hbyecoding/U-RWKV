import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.utils.metrics import iou_score
from src.utils.util import AverageMeter
from src.dataloader.dataset import get_dataloader
from main import get_model, get_args

def evaluate(model, test_loader, device, save_path=None):
    """Evaluate model on test set"""
    model.eval()
    meters = {
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'se': AverageMeter(),
        'pc': AverageMeter(),
        'f1': AverageMeter(),
        'acc': AverageMeter()
    }
    
    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            outputs = model(images)
            iou, dice, se, pc, f1, _, acc = iou_score(outputs, targets)
            
            # Update meters
            meters['iou'].update(iou, images.size(0))
            meters['dice'].update(dice, images.size(0))
            meters['se'].update(se, images.size(0))
            meters['pc'].update(pc, images.size(0))
            meters['f1'].update(f1, images.size(0))
            meters['acc'].update(acc, images.size(0))
            
            # Save predictions if path is provided
            if save_path:
                for i in range(images.size(0)):
                    result = {
                        'prediction': outputs[i].cpu().numpy(),
                        'target': targets[i].cpu().numpy(),
                        'metrics': {
                            'iou': iou,
                            'dice': dice,
                            'se': se,
                            'pc': pc,
                            'f1': f1,
                            'acc': acc
                        }
                    }
                    results.append(result)
    
    # Print results
    print('\nTest Results:')
    print(f'IoU: {meters["iou"].avg:.4f}')
    print(f'Dice: {meters["dice"].avg:.4f}')
    print(f'Sensitivity: {meters["se"].avg:.4f}')
    print(f'Precision: {meters["pc"].avg:.4f}')
    print(f'F1 Score: {meters["f1"].avg:.4f}')
    print(f'Accuracy: {meters["acc"].avg:.4f}')
    
    # Save results if path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics = {
            'iou': meters['iou'].avg,
            'dice': meters['dice'].avg,
            'se': meters['se'].avg,
            'pc': meters['pc'].avg,
            'f1': meters['f1'].avg,
            'acc': meters['acc'].avg
        }
        np.save(save_path / 'metrics.npy', metrics)
        
        # Save predictions
        np.save(save_path / 'predictions.npy', results)
    
    return meters

def main():
    # Get arguments
    parser = argparse.ArgumentParser(description='U-RWKV Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--save-dir', type=str, default='results',
                      help='Directory to save results')
    args = parser.parse_args()
    
    # Load training args from checkpoint
    checkpoint = torch.load(args.checkpoint)
    train_args = checkpoint['args']
    
    # Create model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(train_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test dataloader
    _, test_loader = get_dataloader(
        dataset_name=train_args['dataset'],
        batch_size=1,  # Use batch size 1 for testing
        img_size=train_args['img_size']
    )
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = Path(args.save_dir) / f"{train_args['model']}_{train_args['dataset']}_{timestamp}"
    
    # Evaluate
    meters = evaluate(model, test_loader, device, save_path)
    
    # Save configuration
    config = {
        'model': train_args['model'],
        'dataset': train_args['dataset'],
        'checkpoint': args.checkpoint,
        'timestamp': timestamp
    }
    np.save(save_path / 'config.npy', config)

if __name__ == '__main__':
    main() 