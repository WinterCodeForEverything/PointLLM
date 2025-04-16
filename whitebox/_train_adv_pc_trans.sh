# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference
CUDA_VISIBLE_DEVICES=1 \
python _train_adv_pc_trans.py \
    --model_name 'RunsenXu/PointLLM_7B_v1.2' \
    --data_path '/mnt/ssd/liuchao/PointLLM/Objaverse_npy' \
    --ori_anno_path '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_val_200_GT.json' \
    --adv_anno_path '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_adv_200.json' \
    --output_pc_path '/mnt/ssd/liuchao/PointLLM/Objaverse_adv_npy_coord_0.001_color_0.05' \
    --output_visual_pc_path '/home/liuchao/PointLLM/result/pointcloud' \
    --coord_budget 0.001 \
    --color_budget 0.05 \
    --coord_lr 0.00001 \
    --color_lr 0.0005 \
    --batch_size 16 \
    --num_samples 200 \
    --steps 2000 \
    --use_color \
    --wandb \
    --wandb_project_name 'PointLLM' \
    --wandb_run_name 'eva_adv_coord_0.001_color_0.05_lr_5' \
