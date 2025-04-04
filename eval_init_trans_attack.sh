python pointllm/eval/eval_objaverse.py \
    --data_path /mnt/ssd/liuchao/PointLLM/Objaverse_npy \
    --anno_path /mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_val_200_GT.json \
    --model_name RunsenXu/PointLLM_7B_v1.2 \
    --output_dir ./result \
    --task_type classification \
    --prompt_index 0