# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference
CUDA_VISIBLE_DEVICES=0 \
python _train_trans_and_query_fixed_budget.py \
    --model_name 'RunsenXu/PointLLM_7B_v1.2' \
    --ori_data_path '/mnt/ssd/liuchao/PointLLM/Objaverse_npy' \
    --adv_data_path '/mnt/ssd/liuchao/PointLLM/Objaverse_adv_npy_epsilon_0.05' \
    --ori_anno_path '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_val_200_GT.json' \
    --adv_anno_path '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_adv_200.json' \
    --mrg_output_file '/home/liuchao/PointLLM/result/noise_size_5_result/PointLLM_brief_description_mrg_200_Objaverse_classification_prompt0.json' \
    --output_showing_pc_path '/home/liuchao/PointLLM/result/pointcloud' \
    --output_pc_path '/mnt/ssd/liuchao/PointLLM/Objaverse_pp_tt_adv_npy' \
    --epsilon 0.1 \
    --alpha 0.01 \
    --batch_size 1 \
    --num_samples 200 \
    --num_query 100 \
    --num_sub_query 25 \
    --steps 8 \
    --use_color \
    --max_caption_length 256 \
    --wandb \
    --wandb_project_name 'PointLLM' \
    --wandb_run_name 'eva_pp+tt_noise_size_10' 
