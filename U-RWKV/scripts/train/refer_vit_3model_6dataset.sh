#!/bin/bash

# 定义模型列表
models=("v_enc_256_fffse_dec_fusion_vit" "v_enc_512_fffse_dec_fusion_vit" "v_enc_384_fffse_dec_fusion_vit")

# 定义数据集名称列表
datasets=("poly_1" "clinicDB_4" "colonDB_4" "busi_3" "isic18_3" "isic19") # "clinicDB_4" "colonDB_4" "busi_3" "isic18_3" "isic19"

# 定义日志目录


# 遍历每个模型和数据集的组合
for model in "${models[@]}"; do
  log_dir="./cL${model}logs"
  mkdir -p "$log_dir"
  for dataset in "${datasets[@]}"; do
    # 设置 CUDA_VISIBLE_DEVICES
    echo "$model $dataset"
    CUDA_VISIBLE_DEVICES=0 python Tan9/main_layer_vit_mamba.py \
      --dataset-name "$dataset" \
      --batch-size 8 \
      --epoch 280 \
      --model "$model"
    
    # 检查上一个命令是否成功执行
    if [ $? -ne 0 ]; then
      echo "Error occurred while running model: $model with dataset: $dataset" > "$log_file"
      exit 1
    fi
    
    echo "Finished running model: $model with dataset: $dataset" > "$log_file"
  done
done

echo "All tasks completed successfully!"