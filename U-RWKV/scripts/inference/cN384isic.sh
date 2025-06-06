#!/bin/bash

# 公共参数定义
log_dir="./cN384logs"
mkdir -p "$log_dir"
log_file='terminal_result.txt'
cuda_device=1  # 统一使用 CUDA_VISIBLE_DEVICES=1

# ========== 第二阶段：串行任务 ========== 
# echo -e "\n===== 开始串行任务 =====" > "$log_file"
datasets_serial=("isic18_3" "isic19")

for dataset in "${datasets_serial[@]}"; do
    echo "[串行] 启动 $dataset | 时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 调整批次大小为16
    # 执行命令并捕获输出CUDA_VISIBLE_DEVICES=$cuda_device 
    result=$(CUDA_VISIBLE_DEVICES=2 python Tan9/main_vs_comp_cmunextN.py --dataset-name "$datasets_serial" --batch-size 16 --epoch 280 --model v_enc_384_fffse_dec_resi2_rwkv_with2x4)
    
    # 将结果保存到日志文件
    log_file="${log_dir}/${dataset}_metrics.txt"
    echo "[串行] 完成 $dataset | $result " > "$log_file"
    # # 状态检查
    # if [ $? -eq 0 ]; then
    #         echo "[串行] 完成 $dataset | $result " > "$log_file"
    # else
    #     echo "[串行] 完成 $dataset | 状态: 失败" > "$log_file"
    # fi
    
    echo -e "----------------------------"
done

echo "===== 所有任务执行完毕 ====="

