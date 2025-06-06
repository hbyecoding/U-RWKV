CUDA_VISIBLE_DEVICES=7 python Tan9/main_vit.py --dataset-name busi_3 --model conv_gated_net_randinit_SelfGate --batch-size 4     
CUDA_VISIBLE_DEVICES=7 python Tan9/main_LoRD.py --dataset-name busi_3 --model conv_gated_net_randinit_SelfGate --batch-size 4     
CUDA_VISIBLE_DEVICES=7 python Tan9/main_LoRD_cp.py --dataset-name poly_1 --model compnet_ --batch-size 8 --epoch 50
CUDA_VISIBLE_DEVICES=7 python Tan9/main_LoRD.py --dataset-name poly_1 --model CMUNeXt --batch-size 8 --epoch 50
CUDA_VISIBLE_DEVICES=7 python Tan9/main_LoRD.py --dataset-name poly_1 --model CMUNeXt_2_4_384 --batch-size 8 --epoch 100
CUDA_VISIBLE_DEVICES=3 python Tan9/main_LoRD_cp.py --dataset-name poly_1 --model lord_Brwkv_1222 --batch-size 8 --epoch 100
CUDA_VISIBLE_DEVICES=3 python Tan9/main_LoRD_cp.py --dataset-name poly_1 --model lord_Brwkv_1222 --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name poly_1 --model lord_Brwkv_1222 --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_3 --model lord_Brwkv_1222 --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_3 --model lord_Brwkv_1222 --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main_LoRD.py --dataset-name poly_1 --model CMUNeXt --batch-size 8 --epoch 500
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --model LoRA_DW_4_5 --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name busi_3 --model LoRA_DW_4_5 --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --model CMUNeXt --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main_LoRD.py --dataset-name poly_1 --model CMUNeXt_2_4_9_384 --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main_LoRD_cp.py --dataset-name poly_1 --model LoRA_2_4_9_384Enc --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main_LoRD_cp.py --dataset-name poly_1 --model LoRA_2_4_9_384woDW --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main_LoRD_cp.py --dataset-name poly_1 --model LoRA_wo_dw --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main_LoRD_cp.py --dataset-name poly_1 --model LoRA_2_4_9_384 --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main_LoRD_cp.py --dataset-name poly_1 --model LoRA_4_5_3_4_9_384 --batch-size 8 --epoch 280
CUDA_VISIBLE_DEVICES=5 python Tan9/main_LoRD_cp.py --dataset-name busi_3 --model LoRA_4_5_3_4_9_384 --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=5 python Tan9/main_LoRD_cp.py --dataset-name busi_3 --model LoRA_4_5_qshift --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main_LoRD_cp.py --dataset-name busi_3 --model LoRA__5_qshift --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name busi_3 --model ConvR_Dual --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name busi_2 --model ConvR_Dual --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name busi_1 --model ConvR_Dual --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --model ConvR_Dual --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --model LoRA_4_5_rwkv_ffn_first --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --model LoRA_4_5_dec --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_1 --model LoRA_4_5_dec --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_3 --model LoRA_4_5_dec --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_2 --model LoRA_4_5_dec --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=5 python feature_map/cmunext_rwkv_test.py 
CUDA_VISIBLE_DEVICES=7 python Tan9/src/network/conv_based/re-rwkv.py
CUDA_VISIBLE_DEVICES=4 python Tan9/main_cmunextNconv.py --dataset-name busi_3 --batch-size 4 --epoch 280
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model LoRA_3_4_5
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model LoRA_4_5
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_1 --batch-size 4 --epoch 280 --model LoRA_4_5
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_2 --batch-size 4 --epoch 280 --model LoRA_4_5
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_4_5
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model LoRA_dw_4_5
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_dw_4_5
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model LoRA_dw_rwkv_first_4_5
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model LoRA__5
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA__5
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model CMUNeXt_2_4_9_192_384
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model CMUNeXt_2_4_9_192_384
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_rwkv_2_4_9_192_384
CUDA_VISIBLE_DEVICES=7 python Tan9/main_cmunextNconv.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model compnet_wo_dw1227
CUDA_VISIBLE_DEVICES=5 python Tan9/main_cmunextNconv.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model 
CUDA_VISIBLE_DEVICES=6 python Tan9/main_cmunextNconv.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model CMUNeXt
CUDA_VISIBLE_DEVICES=6 python Tan9/main_cmunextNconv.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model CMUNeXt_downsample_DWT
CUDA_VISIBLE_DEVICES=6 python Tan9/main_cmunextNconv.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model CMUNeXt_downsample_DWT
CUDA_VISIBLE_DEVICES=6 python Tan9/main_cmunextNconv.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model CMUNeXt_downsample_DWT
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model LoRA__5_DWT
CUDA_VISIBLE_DEVICES=2 python Tan9/main.py --dataset-name busi_2 --batch-size 4 --epoch 280 --model LoRA__5_DWT
CUDA_VISIBLE_DEVICES=1 python Tan9/main.py --dataset-name busi_1 --batch-size 4 --epoch 280 --model LoRA__5_DWT
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA__5_DWT
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA__5_downsample
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model LoRA__5_downsample
CUDA_VISIBLE_DEVICES=6 python Tan9/main_cmunextNconv.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model AttU_Net
CUDA_VISIBLE_DEVICES=0 python Tan9/main_cmunextNconv.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model AttU_Net
CUDA_VISIBLE_DEVICES=7 python Tan9/main_cmunextNconv.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=7 python Tan9/main_cmunextNconv.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=6 python Tan9/main_cmunextNconv.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA__5
CUDA_VISIBLE_DEVICES=1 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA__5_DWT
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA_4_5
CUDA_VISIBLE_DEVICES=3 python Tan9/src/
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name isic18_3 --batch-size 4 --epoch 280 --model LoRA__5
CUDA_VISIBLE_DEVICES=6 python Tan9/mablation.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA__5_downsample --ablation GSC
CUDA_VISIBLE_DEVICES=3 python Tan9/mablation.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model DWConvFusionNet --ablation woGSC --dims 24_48_96_192_384
CUDA_VISIBLE_DEVICES=3 python Tan9/mablation.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model DWConvFusionNet --ablation woWT --dims 24_48_96_192_384
CUDA_VISIBLE_DEVICES=2 python Tan9/mablation.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model DWConvFusionNet --ablation WT --dims 24_48_96_192_384
python Tan9/main_retina.py --base-dir ./Tan9/data/DRIVE/ --dataset-name drive --model LoRA_4_5




LoRA_4_5_nlayer8
## Exp_comparion
CUDA_VISIBLE_DEVICES=7 python Tan9/main_cmunextNconv.py --dataset-name busi_2 --batch-size 4 --epoch 280 --model AttU_Net
CUDA_VISIBLE_DEVICES=7 python Tan9/main_cmunextNconv.py --dataset-name busi_1 --batch-size 4 --epoch 280 --model AttU_Net
CUDA_VISIBLE_DEVICES=4 python Tan9/main_cmunextNconv.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=4 python Tan9/main_cmunextNconv.py --dataset-name busi_2 --batch-size 4 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=4 python Tan9/main_cmunextNconv.py --dataset-name busi_1 --batch-size 4 --epoch 280 --model UNeXt

CUDA_VISIBLE_DEVICES=5 python Tan9/main_cmunextNconv.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model DWConvFusionNet --dims 16_32_128_160_256

# Expand
CUDA_VISIBLE_DEVICES=4 python Tan9/main_expand.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_4_5 --dims 48_96_192_384_768
CUDA_VISIBLE_DEVICES=4 python Tan9/main_expand.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_4_5 --dims 48_96_192_384_768
CUDA_VISIBLE_DEVICES=4 python Tan9/main_expand.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_4_5_woBrwkv --dims 48_96_192_384_768
CUDA_VISIBLE_DEVICES=7 python Tan9/main_expand.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA_4_5_nlayer8 --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA_4_5_nlayer8 --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_4_5_nlayer8 --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_4_5_nlayer8_woBrwkv --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA_4_5_nlayer8_woBrwkv --dims 16_32_128_160_256

CUDA_VISIBLE_DEVICES=4 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA_4_5_woBrwkv --dims 48_96_192_384_768
CUDA_VISIBLE_DEVICES=1 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA_4_5_woBrwkv --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=2 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA_4_5_woBrwkv --dims 24_48_96_192_384
CUDA_VISIBLE_DEVICES=4 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_4_5_woBrwkv --dims 48_96_192_384_768
CUDA_VISIBLE_DEVICES=1 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_4_5_woBrwkv --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=2 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LoRA_4_5_woBrwkv --dims 24_48_96_192_384

CUDA_VISIBLE_DEVICES=1 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model LoRA_4_5_woBrwkv --dims 32_64_128_256_512
CUDA_VISIBLE_DEVICES=1 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model DWConvFusionNet --dims 8_16_32_64_128
CUDA_VISIBLE_DEVICES=1 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model DWConvFusionNet --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=2 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model DWConvFusionNet --dims 32_64_128_256_512

CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model DWConvFusionNet --dims 8_16_32_64_128
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model DWConvFusionNet --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model DWConvFusionNet --dims 32_64_128_256_512
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model DWConvFusionNet --dims 32_64_128_256_512

CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model DWConvFusionNet --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model DWConvFusionNet --dims 8_16_32_64_128
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model DWConvFusionNet --dims 8_16_32_64_128
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model DWConvFusionNet --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model TransUnet
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model UNet3plus
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model UNetplus
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LORDDwoWT --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LORDDwoWT --dims 8_16_32_64_128
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LORDDwoWT --dims 32_64_128_256_512
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model UNext yes
CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LORDDwoWT --dims 32_64_128_256_512
CUDA_VISIBLE_DEVICES=6 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model TransUNet
CUDA_VISIBLE_DEVICES=3 python Tan9/main.py --dataset-name busi_1 --batch-size 4 --epoch 280 --model TransUnet
CUDA_VISIBLE_DEVICES=3 python Tan9/main.py --dataset-name busi_2 --batch-size 4 --epoch 280 --model TransUnet


CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model CMUNet




CUDA_VISIBLE_DEVICES=5 python Tan9/main.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model UNext yes
CUDA_VISIBLE_DEVICES=4 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model UNext
CUDA_VISIBLE_DEVICES=7 python Tan9/main.py --dataset-name busi_2 --batch-size 4 --epoch 280 --model U_Net160256


CUDA_VISIBLE_DEVICES=2 python Tan9/main.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model UNet3plus

2025年1月19日
CUDA_VISIBLE_DEVICES=3 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LORDDwoWT --dims 16_32_128_160_256
CUDA_VISIBLE_DEVICES=3 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LORDDwoWT --dims 8_16_32_64_128
CUDA_VISIBLE_DEVICES=3 python Tan9/main.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model LORDDwoWT --dims 32_64_128_256_512


DWConvFusionNet


02-08


CUDA_VISIBLE_DEVICES=0 python Tan9/main_cmunextNconv.py --dataset-name isic19 --batch-size 16 --epoch 280 --model CMUNet
CUDA_VISIBLE_DEVICES=0 python Tan9/main_cmunextNconv.py --dataset-name isic19 --batch-size 16 --epoch 280 --model CMUNet
conda activate rwkv
CUDA_VISIBLE_DEVICES=1 python Tan9/main_cmunextNconv.py --dataset-name isic19 --batch-size 32 --epoch 280 --model AttU_Net

conda activate rwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_cmunextNconv.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model AttU_Net
conda activate rwkv
CUDA_VISIBLE_DEVICES=5 python Tan9/main_cmunextNconv.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model AttU_Net
conda activate rwkv
CUDA_VISIBLE_DEVICES=5 python Tan9/main_cmunextNconv.py --dataset-name colonDB_4 --batch-size 16 --epoch 280 --model U_Net
conda activate rwkv
CUDA_VISIBLE_DEVICES=2 python Tan9/main_cmunextNconv.py --dataset-name isic19 --batch-size 16 --epoch 280 --model U_Net

CUDA_VISIBLE_DEVICES=1 python Tan9/main_cmunextNconv.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=1 python Tan9/main_cmunextNconv.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model UNeXt
conda activate rwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_cmunextNconv.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model UNeXt

conda activate rwkv
CUDA_VISIBLE_DEVICES=5 python Tan9/main_cmunextNconv.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model UNeXt

conda activate rwkv
CUDA_VISIBLE_DEVICES=6 python Tan9/main_cmunextNconv.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model AttU_Net
conda activate rwkv
CUDA_VISIBLE_DEVICES=1 python Tan9/main_cmunextNconv.py --dataset-name poly_1 --batch-size 4 --epoch 2 --model ConvUNeXt

CUDA_VISIBLE_DEVICES=1 python Tan9/main_cmunextNconv.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model ConvUNeXt
CUDA_VISIBLE_DEVICES=1 python Tan9/main_cmunextNconv.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model ConvUNeXt
CUDA_VISIBLE_DEVICES=5 python Tan9/main_cmunextNconv.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model ConvUNeXt
CUDA_VISIBLE_DEVICES=1 python Tan9/main_cmunextNconv.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model ConvUNeXt
CUDA_VISIBLE_DEVICES=2 python Tan9/main_cmunextNconv.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model ConvUNeXt
CUDA_VISIBLE_DEVICES=3 python Tan9/main_cmunextNconv.py --dataset-name isic19 --batch-size 16 --epoch 280 --model ConvUNeXt


CUDA_VISIBLE_DEVICES=7 python Tan9/main_transu_medT_swinu.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model TransUnet
CUDA_VISIBLE_DEVICES=1 python Tan9/main_transu_medT_swinu.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model TransUnet
CUDA_VISIBLE_DEVICES=5 python Tan9/main_transu_medT_swinu.py --dataset-name isic19 --batch-size 1 --epoch 280 --model TransUnet
CUDA_VISIBLE_DEVICES=5 python Tan9/main_transu_medT_swinu.py --dataset-name isic19 --batch-size 8 --epoch 280 --model MedT
CUDA_VISIBLE_DEVICES=5 python Tan9/main_transu_medT_swinu.py --dataset-name isic18_3 --batch-size 8 --epoch 280 --model MedT

dims=[16, 32, 128, 160, 256]
plt erf
CUDA_VISIBLE_DEVICES=2 python Tan9/src/network/conv_based/visualize_erf0109.py 
class ProProcessWithDWConv(nn.Module):
    def __init__(self, ch_in, ch_out, depth=2, k=3):
        """
        Args:
            ch_in (int): 输入通道数
            ch_out (int): 输出通道数
            depth (int): 重复的深度（下采样次数）
            k (int): 深度可分离卷积的核大小
        """
        super(ProProcessWithDWConv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth = depth
        self.k = k

        # 定义下采样模块
        self.downsample_layers = nn.Sequential(
            *[self._make_downsample_layer(ch_in if i == 0 else ch_out) for i in range(depth)]
        )

    def _make_downsample_layer(self, ch):
        """
        创建一个下采样层，包含深度可分离卷积和逐点卷积
        """
        return nn.Sequential(
            # 深度可分离卷积
            nn.Conv2d(ch, ch, kernel_size=(self.k, self.k), stride=2, padding=(self.k // 2, self.k // 2), groups=ch),
            nn.GELU(),
            nn.BatchNorm2d(ch),
            # 逐点卷积
            nn.Conv2d(ch, self.ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(self.ch_out)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (BS, C, H, W)
        Returns:
            torch.Tensor: 输出张量，形状为 (BS, C, H//(2^depth), W//(2^depth))
        """
        return self.downsample_layers(x)


        """fusion"""    
        self.withDW_by_2 = ProProcessWithDWConv(ch_in=self.nemb_by_2, ch_out=self.nemb,depth=1)
        self.withDW_by_4 = ProProcessWithDWConv(ch_in=self.nemb_by_4, ch_out=self.nemb,depth=2)


        # bot2_conv: 一次 Conv2d + BatchNorm2d + GELU
        self.bot2_conv = nn.Sequential(
            nn.Conv2d(self.nemb * 2, self.nemb, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nemb),
            nn.GELU()
        )

        # bot4_conv: 两次 Conv2d + BatchNorm2d + GELU
        self.bot4_conv = nn.Sequential(
            nn.Conv2d(self.nemb * 4, self.nemb *2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nemb* 2),
            nn.GELU(),
            nn.Conv2d(self.nemb*2, self.nemb, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.nemb),
            nn.GELU()
        )
        self.output1 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
        self.output2 = nn.Conv2d(16, num_classes, kernel_size=1, padding=0)
