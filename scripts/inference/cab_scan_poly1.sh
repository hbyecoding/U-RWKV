#!/bin/bash

# 定义模型和数据集列表
models=("abscan_256_fffse_fusion_rwkv_with2x4_1" \
        "abscan_256_fffse_fusion_rwkv_with2x4_2" \
        "abscan_256_fffse_fusion_rwkv_with2x4_1_2" \
        "abscan_256_fffse_fusion_rwkv_with2x4_1_3" \
        "abscan_256_fffse_fusion_rwkv_with2x4_2_4" \
        "abscan_256_fffse_fusion_rwkv_with2x4_3_4" \
        "abscan_256_fffse_fusion_rwkv_with2x4_4" \
        "abscan_256_fffse_fusion_rwkv_with2x4_3")

datasets=("poly_1")  # 数据集列表
BS=4  # 批量大小
num_gpus=1  # 使用的 GPU 数量（假设只用一张卡）

# 将模型列表分为两组
group1=("${models[@]:0:4}")  # 第一组：
group2=("${models[@]:4:4}")  # 第二组：

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
  echo "Processing dataset: $dataset"

  # 运行第一组模型
  echo "Running group 1 models..."
  model_pids=()  # 存储进程 ID 的数组

  for model in "${group1[@]}"; do
    log_dir="./logs/${dataset}_${model}"  # 创建唯一的日志目录
    mkdir -p "$log_dir"
    log_file="${log_dir}/output.log"  # 创建唯一的日志文件

    (
      echo "Starting $model on dataset $dataset with CUDA_VISIBLE_DEVICES=$num_gpus"
      start_time=$(date +%s)  # 记录开始时间

      CUDA_VISIBLE_DEVICES=$num_gpus python Tan9/main_vs_comp_cmunextN.py \
        --dataset-name "$dataset" \
        --batch-size $BS \
        --epoch 280 \
        --model "$model" > "$log_file" 2>&1

      end_time=$(date +%s)  # 记录结束时间
      elapsed_time=$((end_time - start_time))  # 计算耗时

      echo "Eval time for model $model on dataset $dataset: $elapsed_time seconds" >> "$log_file"

      if [ $? -ne 0 ]; then
        echo "Error occurred while running model: $model with dataset: $dataset" >> "$log_file"
      else
        echo "Finished running model: $model with dataset: $dataset" >> "$log_file"
      fi
    ) &

    model_pids+=($!)  # 添加到进程 ID 数组
  done

  # 等待第一组所有模型完成
  for pid in "${model_pids[@]}"; do
    wait "$pid"
  done

  echo "All models in group 1 finished."

  # 运行第二组模型
  echo "Running group 2 models..."
  model_pids=()  # 清空进程 ID 数组

  for model in "${group2[@]}"; do
    log_dir="./logs/${dataset}_${model}"  # 创建唯一的日志目录
    mkdir -p "$log_dir"
    log_file="${log_dir}/output.log"  # 创建唯一的日志文件

    (
      echo "Starting $model on dataset $dataset with CUDA_VISIBLE_DEVICES=$num_gpus"
      start_time=$(date +%s)  # 记录开始时间

      CUDA_VISIBLE_DEVICES=$num_gpus python Tan9/main_vs_comp_cmunextN.py \
        --dataset-name "$dataset" \
        --batch-size $BS \
        --epoch 280 \
        --model "$model" > "$log_file" 2>&1

      end_time=$(date +%s)  # 记录结束时间
      elapsed_time=$((end_time - start_time))  # 计算耗时

      echo "Eval time for model $model on dataset $dataset: $elapsed_time seconds" >> "$log_file"

      if [ $? -ne 0 ]; then
        echo "Error occurred while running model: $model with dataset: $dataset" >> "$log_file"
      else
        echo "Finished running model: $model with dataset: $dataset" >> "$log_file"
      fi
    ) &

    model_pids+=($!)  # 添加到进程 ID 数组
  done

  # 等待第二组所有模型完成
  for pid in "${model_pids[@]}"; do
    wait "$pid"
  done

  echo "All models in group 2 finished."
done

echo "All tasks completed successfully!"