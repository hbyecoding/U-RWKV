# 训练完整模型
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model compnet_full
#no->ok out = model()[0]
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model compnet_full
#no->ok out = model()[0]

CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model compnet_full
#ok

# 训练移除 SELayer 的模型
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model compnet_no_se
#ok
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model compnet_no_se
#ok
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model compnet_no_se
#ok

# 训练移除 Autoencoder Attention Map 的模型
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model compnet_no_attention
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model compnet_no_attention
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model compnet_no_attention

# oks for 2 on 3
#oks for 4 on 1


# 训练仅保留 Segmentation Decoder 的模型
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model compnet_single_decoder
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model compnet_single_decoder
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model compnet_single_decoder
# oks for 3 on 5


# 训练减少 Encoder 深度的模型
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model compnet_shallow_encoder
#ok
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model compnet_shallow_encoder
#ok
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model compnet_shallow_encoder
#ok
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model compnet_add_encoder
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model compnet_add_encoder
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model compnet_add_encoder


CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model CMUNet
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model CMUNet
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model CMUNet 
## (但应该没什么用) over


CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext
#ok over

CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext
# oks


CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_gsc
#ok
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_gsc
#ok
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_gsc
#ok

CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_gsc_no_pm
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_gsc_no_pm
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_gsc_no_pm
#ok

##
python main.py --model MedT --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt --base_lr 0.01 --epoch 300 --batch_size 8
# 0122 AM1039
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_no_chin_4
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_no_chin_4
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_no_chin_4
#ok
# 0122 AM1043
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_no_chout_4_fusion
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_no_chout_4_fusion
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_no_chout_4_fusion
#ok

CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_no_chin_4_withSE_4
#ok
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_no_chin_4_withSE_4
#ok
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model  cmunext_no_chin_4_withSE_4
#ok

CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model  cmunext_no_chin_4_withSE_16
#ok
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model  cmunext_no_chin_4_withSE_16
#ok
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model   cmunext_no_chin_4_withSE_16
#ok

CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model  cmunext_no_chin_4_withSE_8
#ok
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model  cmunext_no_chin_4_withSE_8
#ok
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model   cmunext_no_chin_4_withSE_8
#ok
# rwkv 有几个要 改进的 点。

cmunext_（带chin4)

CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model comp_rwkv
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model comp_rwkv
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model comp_rwkv


CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model comp_rwkv_5
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model comp_rwkv_5
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model comp_rwkv_5

# ok

CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_chin_4_no_fusion_dec_se4
#ok
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_chin_4_no_fusion_dec_se4
#ok
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model  cmunext_chin_4_no_fusion_dec_se4
#ok

CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_chin_4_no_fusion_dec_se8
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_chin_4_no_fusion_dec_se8
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model  cmunext_chin_4_no_fusion_dec_se8
#ok

CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_chin_4_no_fusion_dec_se_rwkv5
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_chin_4_no_fusion_dec_se_rwkv5
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model  cmunext_chin_4_no_fusion_dec_se_rwkv5


CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_chin_4_no_fusion_dec_se_rwkv45
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_chin_4_no_fusion_dec_se_rwkv
#ok
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model  cmunext_chin_4_no_fusion_dec_se_rwkv
#ok

CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model  cmunext_chin_4_no_fusion_dec_se_rwkv
#


CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_rwkv5
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_rwkv45

CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_rwkv5
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_rwkv45

CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_rwkv5
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_rwkv45


CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_rwkv45_



CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_rwkv45

CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_rwkv45

# 垂直向内扫描
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_rwkv45


CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_rwkv45_allchannel_shift

CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_rwkv45_allchannel_shift

CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_rwkv45_allchannel_shift

CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model cmunext_rwkv45_octo_shift

CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model cmunext_rwkv45_octo_shift

CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model cmunext_rwkv45_octo_shift 45 cmunext_rwkv45_octo_shift
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_1 --batch-size 4 --epoch 280 --model cmunext_rwkv
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model cmunext
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model compnet_full
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model comp_rwkv
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model comp_rwkv_5
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model U_Net


CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model cmunext
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model compnet_full
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model comp_rwkv_5
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model comp_rwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model U_Net
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model U_Net


CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model chin4_se_depths1_rwkv
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model chin4_se_depths1_rwkv
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model chin4_se_depths1_rwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 4 --epoch 280 --model chin4_se_depths1_rwkv
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model chin4_se_depths1_rwkv

CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model chin4_DWse_depths_rwkv
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model chin4_DWse_depths_rwkv
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model chin4_DWse_depths_rwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 4 --epoch 280 --model chin4_DWse_depths_rwkv
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model chin4_DWse_depths_rwkv

02-04
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model chin4_se_depths_rwkv
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model chin4_se_depths_rwkv
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model chin4_se_depths_rwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 4 --epoch 280 --model chin4_se_depths_rwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 4 --epoch 280 --model chin4_se_depths_rwkv
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model chin4_se_depths_rwkv


CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 4 --epoch 280 --model chin4_se_depths_rwkv

02-05

conda activate rwkv
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model compnet_single_decoder_encgate_depths_all1
conda activate rwkv
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model compnet_single_decoder_encgate_depths_all1

conda activate rwkv
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model compnet_single_decoder_encgate_depths_all1

conda activate rwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 4 --epoch 280 --model compnet_single_decoder_encgate_depths_all1

conda activate rwkv
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model compnet_single_decoder_encgate_depths_all1


in VM-UNet 
CUDA_VISIBLE_DEVICES=2 python train.py

# for test isic2018
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name isic19 --batch-size 4 --epoch 280 --model compnet_full
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name isic19 --batch-size 4 --epoch 280 --model cmunext
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name isic19 --batch-size 32 --epoch 280 --model cmunext

# 4sacn

CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model comp_rwkv_5_4scan
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model comp_rwkv_5_4scan
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model comp_rwkv_5_4scan
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model comp_rwkv_5_4scan
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model comp_rwkv_5_4scan

CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model comp_rwkv_4scan
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model comp_rwkv_4scan
conda activate rwkv
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model comp_rwkv_4scan
conda activate rwkv
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model comp_rwkv_4scan
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model comp_rwkv_4scan



CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic19 --batch-size 32 --epoch 280 --model comp_rwkv_5_4scan
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic19 --batch-size 32 --epoch 280 --model comp_rwkv_4scan

02-09 先前  dice 出错 所以放弃 convN 这个main
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model ConvUNeXt ok
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model ConvUNeXt ok
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model ConvUNeXt ok
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model ConvUNeXt

CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model ConvUNeXt ok
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model ConvUNeXt


test
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 2 --model UNeXt

CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model UNeXt
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model UNeXt
# oks

CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model U_Net
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model U_Net
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model U_Net
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model U_Net
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model U_Net
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model U_Net

CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model TransUnet

CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model AttU_Net
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model AttU_Net


CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model SwinUnet
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model SwinUnet
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model MedT
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model MedT
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model MedT
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model MedT

CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model CMUNet
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model CMUNet

CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model SwinUnet
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model SwinUnet
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model SwinUnet
未成功，报错说：
(rwkv) (base) hongboye@amax:~/projects$ CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16
 --epoch 280 --model SwinUnet
0-0-0-0-0-0-0- train args get!
PATH isic18_train3.txt ./Tan9/data/isic18/isic18_train3.txt
total 1815  train samples
total 779  val samples
train num:1815, val num:779
usage: main_vs_comp_cmunext.py [-h] [--zip] [--cfg CFG] [--opts OPTS [OPTS ...]] [--cache-mode {no,full,part}] [--resume RESUME]
                               [--accumulation-steps ACCUMULATION_STEPS] [--use-checkpoint] [--amp-opt-level {O0,O1,O2}] [--tag TAG]
                               [--eval] [--throughput]
main_vs_comp_cmunext.py: error: unrecognized arguments: --dataset-name isic18_3 --batch-size 16 --epoch 280 --model comp_rwkv_4scan

CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunext.py --dataset-name isic19 --batch-size 32 --epoch 280 --model comp_rwkv_4scan
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model comp_rwkv_4scan
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunext.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model comp_rwkv_4scan


CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model comp_rwkv_5_4scan
未执行

CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunext.py --dataset-name isic19 --batch-size 32 --epoch 280 --model comp_rwkv_5_4scan
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunext.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model comp_rwkv_5_4scan

CUDA_VISIBLE_DEVICES=3 python main.py --model SwinUnet --img_size 224 --dataset-name busi
CUDA_VISIBLE_DEVICES=7 python main.py --model SwinUnet --img_size 224 --dataset-name isic18 --batch_size 16
待确认。。。
CUDA_VISIBLE_DEVICES=0 python main.py --model SwinUnet --img_size 224 --dataset-name isic19 --batch_size 32

CUDA_VISIBLE_DEVICES=1 python main.py --model SwinUnet --img_size 224 --dataset-name poly
能行

CUDA_VISIBLE_DEVICES=2 python main.py --model SwinUnet --img_size 224 --dataset-name clinicDB 

CUDA_VISIBLE_DEVICES=1 python main.py --model SwinUnet --img_size 224 --dataset-name colonDB
# 能行

CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 8 --epoch 280 --model MedT

CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextA.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model chin4_se_depths_rwkv_5_with4scan
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextA.py --dataset-name isic19 --batch-size 32 --epoch 280 --model chin4_se_depths_rwkv_5_with4scan
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunextA.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model chin4_se_depths_rwkv_5_with4scan
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextA.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model chin4_se_depths_rwkv_5_with4scan
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextA.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model chin4_1x1_se_depths_rwkv_5_with4scan
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextA.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model chin4_1x1_se_depths_rwkv_5_with4scan
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextA.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model chin4_1x1_se_depths_rwkv_5_with4scan
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextA.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model chin4_1x1_se_depths_rwkv_5_with4scan
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextA.py --dataset-name isic19 --batch-size 32 --epoch 280 --model chin4_1x1_se_depths_rwkv_5_with4scan
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunextA.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model chin4_1x1_se_depths_rwkv_5_with4scan


CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model TransUnet
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model TransUnet
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model CMUNet
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model CMUNet

# 训练
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_sym256384
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model testLoRA__5_160256
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model testLoRA__5_160256
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model testLoRA__5_160256

# test tiny  tiny rwkv
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model tinyUnet
# 无法使用
# 


## test fffse
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_simple2
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_simple2
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_simple2
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_simple2


CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_withbirwkv

CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2

CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2x4
已经ok

CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_withbirwkv
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_withbirwkv
你是关于脚本执行包括python项目文件运行的专家 我的诉求是 ：
执行完上面的代码之后， 
检测到 CUDA_VISIBLE_DEVICES= 后面的gpu id 的gpu 卡上无任务的时候， 顺次执行
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 32 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2x4
检测到 CUDA_VISIBLE_DEVICES= 后面的gpu id 的gpu 卡上无任务的时候， 顺次执行
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2x4

CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model v_enc_256_fffse_dec_fusion_rwkv_with2x4
检测到 CUDA_VISIBLE_DEVICES= 后面的gpu id 的gpu 卡上无任务的时候， 顺次执行
CUDA_VISIBLE_DEVICES=0 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model v_enc_256_fffse_dec_fusion_rwkv_with2x4

CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_fusion_rwkv_with2x4



CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_fusion_rwkv_with2xVF
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_fusion_rwkv_with2xVF
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_256_fffse_dec_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_128_fffse_dec_fusion_rwkv_with2xVF


CUDA_VISIBLE_DEVICES=3 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_128_fffse_decx2_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_128_fffse_decx2_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_128_fffse_decx2_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=6 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_128_fffse_decx2_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model v_enc_128_fffse_decx2_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model v_enc_128_fffse_decx2_fusion_rwkv_with2x4


CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model v_enc_256_fffse_dec_fusion_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model v_enc_128_fffse_dec_resi2_rwkv_with2x4

v_enc_384_fffse_dec_fusion_rwkv_with2x4
v_enc_512_fffse_dec_fusion_rwkv_with2x4

v_enc_256_fffse_dec_fusion_with2x4(dims=[32, 64, 128, 256, 512])

CUDA_VISIBLE_DEVICES=5 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 16 --epoch 2 --model tinyUnet
# new sh
--dims 16_32_128_160_256

--dims 8_16_32_64_128
--dims 24_48_96_192_384
--dims 32_64_128_256_512
--dims 48_96_192_384_768
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model v_enc_256_fffse_dec_resi2_rwkv_with2x4 --dims 24_48_96_192_384

CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic18_3 --batch-size 16 --epoch 280 --model v_enc_512_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=4 python Tan9/main_vs_comp_cmunextN.py --dataset-name isic19 --batch-size 32 --epoch 280 --model v_enc_512_fffse_dec_resi2_rwkv_with2x4


CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name poly_1 --batch-size 4 --epoch 280 --model v_enc_512_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name clinicDB_4 --batch-size 4 --epoch 280 --model v_enc_512_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name colonDB_4 --batch-size 4 --epoch 280 --model v_enc_512_fffse_dec_resi2_rwkv_with2x4
CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name busi_3 --batch-size 4 --epoch 280 --model v_enc_512_fffse_dec_resi2_rwkv_with2x4



CUDA_VISIBLE_DEVICES=1 
python Tan9/minfer_vs_comp_cmunextN.py --model tinyUnet

# about vmunet train synapse
conda activate vmunet
CUDA_VISIBLE_DEVICES=3 python train_synapse_modelname.py --modelname ConvUNeXt
# ok
conda activate vmunet
CUDA_VISIBLE_DEVICES=5 python train_synapse_modelname.py --modelname UNeXt
# ok

conda activate vmunet
CUDA_VISIBLE_DEVICES=7 python train_synapse_modelname.py --modelname chin4_se_depths_rwkv_5_with4scan

conda activate vmunet
CUDA_VISIBLE_DEVICES=0 python train_synapse_modelname.py --modelname U_Net
# ok # ok

conda activate vmunet
CUDA_VISIBLE_DEVICES=1 python train_synapse_modelname.py --modelname CMUNet
# ok # ok
conda activate vmunet
CUDA_VISIBLE_DEVICES=2 python train_synapse_modelname.py --modelname AttU_Net
# ok
# 测试一下是不是 
python split.py --dataset_root ./data --dataset_name colonDB
# 过去 cmunext  chin4  decoder
先 up
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
        然后             d5 = self.Up5(x5)
            d5 = torch.cat((x4, d5), dim=1)
            d5 = self.Up_conv5(d5)

也即是：up ch_in, ch_out
         fusion 是 ch_in, ch_in, ;chin chout*4;ch_out * 4, ch_out

## compnet 则是：
class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c+out_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.upsample(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x


