#!/bin/bash

models=("v_enc_768_fffse_dec_fusion_rwkv_with2x4")


datasets=("poly_1" "clinicDB_4" "busi_3") # "clinicDB_4" "colonDB_4" "busi_3" "isic18_3" "isic19""colonDB_4"
BS=4
cudanum=4
# 定义日志目录

# model="tinyUnet"
# 遍历每个模型和数据集的组合
for model in "${models[@]}"; do
  log_dir="./cTest${model}logs"
  mkdir -p "$log_dir"
  for dataset in "${datasets[@]}"; do
    (    # 设置 CUDA_VISIBLE_DEVICES
        log_file="${log_dir}/${model}_${dataset}.log"
        echo "$model $dataset"
        CUDA_VISIBLE_DEVICES=$cudanum python Tan9/main_vs_comp_cmunextN.py --dataset-name "$dataset" --batch-size $BS --epoch 280 --model "$model"
        
        # 检查上一个命令是否成功执行
        if [ $? -ne 0 ]; then
        echo "Error occurred while running model: $model with dataset: $dataset" > "$log_file"
        exit 1
        fi
        
        echo "BS $BS on $cudanum Finished running model: $model with dataset: $dataset" > "$log_file"
    ) &
  done
done

echo "All tasks completed successfully!"