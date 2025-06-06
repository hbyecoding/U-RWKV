# U-RWKV

A medical image segmentation model based on RWKV attention mechanism.

## Project Structure

```
U-RWKV/
├── models/
│   ├── v_enc_256/         # Base model with 256 channels
│   ├── v_enc_384/         # Model variant with 384 channels
│   ├── v_enc_512/         # Model variant with 512 channels
│   ├── v_enc_768/         # Model variant with 768 channels
│   └── components/        # Shared model components
├── train/                 # Training scripts
├── utils/                 # Utility functions
└── configs/              # Configuration files
```

## Model Variants

- v_enc_256_fffse_dec_fusion_rwkv_with2x4 (base version)
- v_enc_384_fffse_dec_fusion_rwkv_with2x4
- v_enc_512_fffse_dec_fusion_rwkv_with2x4
- v_enc_768_fffse_dec_fusion_rwkv_with2x4
- Special variants:
  - with2x4_wo: Without certain components
  - simpchinchout: Simplified channel in/out version

## Training

The main training scripts are:
- main.py: Main training script
- eval.py: Evaluation script

## Dependencies

Required Python packages:
- PyTorch
- torchvision
- einops
- timm
- wandb (for experiment tracking)

## Features

- RWKV-based attention mechanism for efficient sequence processing
- U-Net-like architecture for medical image segmentation
- Support for multiple medical image datasets
- Various model configurations and ablation studies

## Directory Structure

```
U-RWKV/
├── models/
│   ├── cmunext/        # CMUNeXt model implementations
│   ├── components/     # Common model components
│   └── utils/          # Utility functions
├── scripts/
│   ├── train/         # Training scripts
│   └── inference/     # Inference scripts
├── configs/           # Configuration files
└── docs/             # Documentation
```

## Installation

```bash
git clone https://github.com/hbyecoding/U-RWKV.git
cd U-RWKV
pip install -r requirements.txt
```

## Usage
### HERE IS A PRIVATE VERSION ,MAYBE YOU COULD NOT BUILD IT ~
TODO
[ ] add v_enc_rwkv (U-RWKV)
### Training

```bash
bash scripts/train/train.sh
```

### Inference

```bash
bash scripts/inference/infer.sh
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
Coming soon...
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the RWKV community for their pioneering work
- Thanks to all contributors to this project 
