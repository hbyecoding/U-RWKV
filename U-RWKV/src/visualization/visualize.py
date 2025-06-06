import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def visualize_feature_maps(feature_maps, save_dir, layer_name):
    """Visualize feature maps from a specific layer"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy if needed
    if torch.is_tensor(feature_maps):
        feature_maps = feature_maps.detach().cpu().numpy()
    
    # Get dimensions
    n_features = feature_maps.shape[1]
    
    # Create grid layout
    grid_size = int(np.ceil(np.sqrt(n_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    
    # Plot each feature map
    for idx in range(n_features):
        i, j = idx // grid_size, idx % grid_size
        feature_map = feature_maps[0, idx]
        
        # Normalize feature map
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        axes[i, j].imshow(feature_map, cmap='viridis')
        axes[i, j].axis('off')
    
    # Remove empty subplots
    for idx in range(n_features, grid_size * grid_size):
        i, j = idx // grid_size, idx % grid_size
        fig.delaxes(axes[i, j])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{layer_name}_feature_maps.png')
    plt.close()

def visualize_attention_maps(attention_weights, save_dir, name):
    """Visualize attention weights"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Plot attention weights as heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_weights, cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Weights - {name}')
    plt.savefig(save_dir / f'{name}_attention.png')
    plt.close()

def visualize_predictions(image, mask, prediction, save_dir, name):
    """Visualize input image, ground truth mask and prediction"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    
    # Normalize image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot image
    axes[0].imshow(np.transpose(image, (1, 2, 0)))
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Plot ground truth
    axes[1].imshow(mask[0], cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction
    axes[2].imshow(prediction[0], cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{name}_comparison.png')
    plt.close()

def visualize_receptive_field(model, layer_name, input_size=(256, 256), save_dir=None):
    """Visualize effective receptive field of a specific layer"""
    save_dir = Path(save_dir) if save_dir else Path('visualization/receptive_field')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create input with central impulse
    input_tensor = torch.zeros((1, 3, *input_size))
    center = (input_size[0] // 2, input_size[1] // 2)
    input_tensor[0, :, center[0], center[1]] = 1.0
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get feature maps
    feature_maps = output.squeeze().cpu().numpy()
    
    # Visualize receptive field
    plt.figure(figsize=(10, 10))
    plt.imshow(np.mean(feature_maps, axis=0), cmap='viridis')
    plt.colorbar()
    plt.title(f'Effective Receptive Field - {layer_name}')
    plt.savefig(save_dir / f'{layer_name}_receptive_field.png')
    plt.close()

def plot_training_curves(metrics, save_dir):
    """Plot training and validation curves"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_dir / 'loss_curves.png')
    plt.close()
    
    # Plot metric curves
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_iou'], label='Train IoU')
    plt.plot(metrics['val_iou'], label='Val IoU')
    plt.plot(metrics['train_dice'], label='Train Dice')
    plt.plot(metrics['val_dice'], label='Val Dice')
    plt.title('Metric Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(save_dir / 'metric_curves.png')
    plt.close() 