#!/bin/bash

# 定义模型列表
# models=("tinyUnet" "v_enc_512_fffse_dec_fusion_vit" "v_enc_384_fffse_dec_fusion_vit")

# 定义数据集名称列表
datasets=("isic18_3" "isic19") # "clinicDB_4" "colonDB_4" "busi_3" "isic18_3" "isic19"
BS=16
cudanum=4
# 定义日志目录

model="tinyUnet"
# 遍历每个模型和数据集的组合
# for model in "${models[@]}"; do
log_dir="./cL${model}logs"
mkdir -p "$log_dir"
for dataset in "${datasets[@]}"; do
  
  log_file="${log_dir}/${model}_${dataset}.log"
  echo "$model $dataset" | tee -a "$log_file"

  # 设置 CUDA_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=$cudanum
  python Tan9/main_vs_comp_cmunextN.py \
      --dataset-name "$dataset" \
      --batch-size "$BS" \
      --epoch 280 \
      --model "$model" 2>&1 | tee -a "$log_file"

  # 将输出保存到日志文件
  # echo "$result" >> "$log_file"
  
  # 检查上一个命令是否成功执行
  if [ $? -ne 0 ]; then
    echo "Error occurred while running model: $model with dataset: $dataset" | tee -a "$log_file"
    exit 1
  fi
  
  echo "On $cudanum Finished running model: $model with dataset: $dataset BS $BS" | tee -a "$log_file"
done
# done

echo "All tasks completed successfully!"

