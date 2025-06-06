#!/bin/bash

# 定义数据集名称列表
datasets=("poly_1" "clinicDB_4" "colonDB_4" "busi_3")

# # 并行运行任务
# for dataset in "${datasets[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py \
#         --dataset-name "$dataset" \
#         --batch-size 4 \
#         --epoch 2 \
#         --model v_enc_512_fffse_dec_resi2_rwkv_with2x4 > "log_$dataset.txt" 2>&1 &
# done

#!/bin/bash


# 定义日志目录
log_dir="./cN384logs"
mkdir -p "$log_dir"

# 并行运行任务
for dataset in "${datasets[@]}"; do
    (
        result=$(CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py --dataset-name "$dataset" --batch-size 4 --epoch 280 --model v_enc_384_fffse_dec_resi2_rwkv_with2x4)
        
        # 将结果保存到日志文件
        log_file="${log_dir}/${dataset}_metrics.txt"
        echo "$result" > "$log_file"
    ) &
done

# 等待所有任务完成
wait


#!/bin/bash

# 定义数据集名称列表
datasets_2=("isic18_3" "isic19")

# 定义日志目录
log_dir="./cN512logs"
mkdir -p "$log_dir"

# 串行运行任务
for dataset in "${datasets[@]}"; do
    # 显示开始时间
    echo "=== Start processing $dataset at $(date '+%Y-%m-%d %H:%M:%S') ==="
    
    # 执行命令并捕获输出
    CUDA_VISIBLE_DEVICES=1 python Tan9/main_vs_comp_cmunextN.py \
        --dataset-name "$dataset" \
        --batch-size 4 \
        --epoch 280 \
        --model v_enc_512_fffse_dec_resi2_rwkv_with2x4 > "${log_dir}/${dataset}_metrics.txt" 2>&1

    # 检查执行状态
    if [ $? -eq 0 ]; then
        echo "=== Successfully completed $dataset ==="
    else
        echo "!!! Failed to process $dataset !!!"
    fi
    
    # 添加间隔
    echo -e "\n"
done
