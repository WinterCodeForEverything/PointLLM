# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference
CUDA_VISIBLE_DEVICES=0 \
python _train_adv_pc_trans_HiT.py \
    --model_name 'RunsenXu/PointLLM_7B_v1.2' \
    --data_path '/mnt/ssd/liuchao/PointLLM/Objaverse_npy_with_normals' \
    --ori_anno_path '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_val_200_GT.json' \
    --adv_anno_path '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_adv_200.json' \
    --output_pc_path '/mnt/ssd/liuchao/PointLLM/Objaverse_adv_npy_CD_noise_size_5' \
    --use_color \
    --use_normal \
    --batch_size 8 \
    --attack_lr 0.005 \
    --init_weight 0.5 \
    --max_weight 2.0 \
    --success_threshold 0.55 \
    --binary_step 1 \
    --num_iter 1000 \
    --wandb \
    --wandb_project_name 'PointLLM' \
    --wandb_run_name 'eva_HiT_attack' \

    # --epsilon 0.05 \
    # --attack_lr 0.001 \
    # --gamma1 1.0 \
    
    # --num_samples 200 \
    # --steps 2000 \
