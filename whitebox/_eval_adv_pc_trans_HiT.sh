CUDA_VISIBLE_DEVICES=1 \
python _eval_adv_pc_trans_HiT.py \
    --ori_pc_path /mnt/ssd/liuchao/PointLLM/Objaverse_npy_with_normals \
    --adv_pc_path /mnt/ssd/liuchao/PointLLM/Objaverse_adv_npy_coord_0.001_color_0.05 \
    --ori_anno_path /mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_val_200_GT.json \
    --adv_anno_path /mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_adv_200.json \
    --model_name RunsenXu/PointLLM_7B_v1.2 \
    --output_dir /home/liuchao/PointLLM/result/MF-pp_adv_result_coord_0.001_color_0.05 \
    --task_type classification \
    --prompt_index 0 \
    --start_caption_eval \
    --gpt_type gpt-4o-mini \
     --start_pc_eval \