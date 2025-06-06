#!/bin/bash

# 定义数据集名称列表
datasets=("busi_3" "poly_1" "clinicDB_4")

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
log_dir="./cab_h_v_isiclogs"
mkdir -p "$log_dir"

# 并行运行任务
for dataset in "${datasets[@]}"; do
    (
        result=$(CUDA_VISIBLE_DEVICES=7 python Tan9/main_vs_comp_cmunextN.py --dataset-name "$dataset" --batch-size 4 --epoch 280 --model vscan_enc_256_fffse_dec_fusion_rwkv2_h_v)
        
        # 将结果保存到日志文件
        log_file="${log_dir}/${dataset}_metrics.txt"
        echo "$result" > "$log_file"
    ) &
done

# # 等待所有任务完成
# wait