---
language: en
tags:
- medical-image-segmentation
- pytorch
- u-rwkv
datasets:
- rwkv
metrics:
- dice
model-index:
- name: u-rwkv-polyp
  results:
  - task: 
      type: image-segmentation
      name: Medical Image Segmentation
    dataset:
      type: rwkv
      name: RWKV
    metrics:
      - type: dice
        value: 0.7887
---

# u-rwkv-polyp

This model is part of the U-RWKV family of medical image segmentation models. It combines the power of RWKV (Receptance Weighted Key Value) attention mechanism with U-Net architecture for efficient and accurate medical image segmentation.

## Model description

U-RWKV model trained on Polyp dataset for polyp segmentation

### Architecture

- Base architecture: U-Net with RWKV attention
- Input channels: 3
- Output channels: 1
- Base channels: [16, 32, 128, 160, 256]
- Attention mechanism: RWKV (Receptance Weighted Key Value)
- Feature fusion: SE (Squeeze-and-Excitation)

## Performance

- Dice score: 0.7887

## Usage

```python
import torch
from models.model import U_RWKV

# Load model
model = U_RWKV()
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    output = model(input_image)
```

## Training

The model was trained using:
- Loss functions: Dice Loss + BCE Loss
- Optimizer: AdamW
- Learning rate: 1e-4
- Batch size: 16
- Data augmentation: Random flip, rotation, scaling

## License

This model is released under the MIT License.
