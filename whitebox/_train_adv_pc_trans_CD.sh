# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference
CUDA_VISIBLE_DEVICES=0 \
python _train_adv_pc_trans_CD.py \
    --model_name 'RunsenXu/PointLLM_7B_v1.2' \
    --data_path '/mnt/ssd/liuchao/PointLLM/Objaverse_npy' \
    --ori_anno_path '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_val_200_GT.json' \
    --adv_anno_path '/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_adv_200.json' \
    --output_pc_path '/mnt/ssd/liuchao/PointLLM/Objaverse_adv_npy_CD_noise_size_5' \
    --epsilon 0.05 \
    --alpha 0.001 \
    --gamma1 1.0 \
    --batch_size 16 \
    --num_samples 200 \
    --steps 2000 \
    --use_color \
    --wandb \
    --wandb_project_name 'PointLLM' \
    --wandb_run_name 'eva_chamfer_CD_noise_size_5' \
