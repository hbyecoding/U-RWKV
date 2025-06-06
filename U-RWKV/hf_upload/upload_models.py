from huggingface_hub import HfApi, create_repo
import os
import json

# Hugging Face配置
TOKEN = "hf_FLFLGCNzXXXXXXXXXXXXXX"  # 请替换为您的实际token
ORGANIZATION = "Hugoyeah"

# 模型配置
MODELS = {
    "busi": {
        "name": "u-rwkv-busi",
        "description": "U-RWKV model trained on BUSI dataset for breast ultrasound image segmentation",
        "model_path": "best_models/busi_best.pth",
        "metrics": {
            "dice": 0.6973
        }
    },
    "isic18": {
        "name": "u-rwkv-isic18",
        "description": "U-RWKV model trained on ISIC 2018 dataset for skin lesion segmentation",
        "model_path": "best_models/isic18_best.pth",
        "metrics": {
            "dice": 0.8226
        }
    },
    "clinicdb": {
        "name": "u-rwkv-clinicdb",
        "description": "U-RWKV model trained on ClinicDB dataset for polyp segmentation",
        "model_path": "best_models/clinicdb_best.pth",
        "metrics": {
            "dice": 0.8596
        }
    },
    "polyp": {
        "name": "u-rwkv-polyp",
        "description": "U-RWKV model trained on Polyp dataset for polyp segmentation",
        "model_path": "best_models/polyp_best.pth",
        "metrics": {
            "dice": 0.7887
        }
    }
}

def create_model_card(model_info):
    """Create a model card for the given model"""
    return f"""---
language: en
tags:
- medical-image-segmentation
- pytorch
- u-rwkv
datasets:
- {model_info["name"].split("-")[1]}
metrics:
- dice
model-index:
- name: {model_info["name"]}
  results:
  - task: 
      type: image-segmentation
      name: Medical Image Segmentation
    dataset:
      type: {model_info["name"].split("-")[1]}
      name: {model_info["name"].split("-")[1].upper()}
    metrics:
      - type: dice
        value: {model_info["metrics"]["dice"]}
---

# {model_info["name"]}

This model is part of the U-RWKV family of medical image segmentation models. It combines the power of RWKV (Receptance Weighted Key Value) attention mechanism with U-Net architecture for efficient and accurate medical image segmentation.

## Model description

{model_info["description"]}

### Architecture

- Base architecture: U-Net with RWKV attention
- Input channels: 3
- Output channels: 1
- Base channels: [16, 32, 128, 160, 256]
- Attention mechanism: RWKV (Receptance Weighted Key Value)
- Feature fusion: SE (Squeeze-and-Excitation)

## Performance

- Dice score: {model_info["metrics"]["dice"]}

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
"""

def upload_model(model_key, model_info):
    """Upload a model to the Hugging Face Hub"""
    api = HfApi()
    repo_id = f"{ORGANIZATION}/{model_info['name']}"
    
    # Create and upload model card
    model_card = create_model_card(model_info)
    with open("README.md", "w") as f:
        f.write(model_card)
    
    # Upload files
    try:
        print(f"Uploading files to {repo_id}...")
        
        # Upload model card
        print("Uploading README.md...")
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            token=TOKEN
        )
        
        # Upload model file
        print(f"Uploading {model_info['model_path']}...")
        api.upload_file(
            path_or_fileobj=model_info["model_path"],
            path_in_repo="model.pth",
            repo_id=repo_id,
            token=TOKEN
        )
        
        print(f"Successfully uploaded {model_key} model to {repo_id}")
    except Exception as e:
        print(f"Upload failed for {model_key}: {e}")

def main():
    for model_key, model_info in MODELS.items():
        print(f"Uploading {model_key} model...")
        upload_model(model_key, model_info)

if __name__ == "__main__":
    main() 