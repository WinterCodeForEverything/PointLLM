#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
#from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.data import ObjectPointCloudDataset
from tqdm import tqdm
from transformers import AutoTokenizer
# from pointllm.eval.evaluator import start_evaluation
import os
import json
import wandb

from pointllm.eval.evaluator import start_evaluation

from whitebox.evaluation.pc_evaluator.pc_evaluator import PointCloudEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"

def init_pc_encoder(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    pc_encoder = model.get_pc_encoder()
    
    return pc_encoder

def load_dataset(data_path, anno_path, pointnum, conversation_types, use_color):
    print("Loading validation datasets.")
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pointnum=pointnum,
        conversation_types=conversation_types,
        use_color=use_color,
        tokenizer=None # * load point cloud only
    )
    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def pc_norm(pc):
    """Normalize the first 3 dimensions of the point cloud to range [0, 1].
    Args:
        pc: torch.Tensor of shape (B, N, 6), where B is batch size, N is number of points.
    Returns:
        torch.Tensor of shape (B, N, 6) with the first 3 dimensions normalized to range [0, 1].
    """
    coords = pc[:, :, :3]  # Extract the first 3 dimensions
    min_coords = coords.min(dim=1, keepdim=True).values  # (B, 1, 3)
    max_coords = coords.max(dim=1, keepdim=True).values  # (B, 1, 3)
    normalized_coords = (coords - min_coords) / (max_coords - min_coords + 1e-8)  # Avoid division by zero
    pc[:, :, :3] = normalized_coords  # Replace the first 3 dimensions with normalized values
    return pc

def main(args):
    
    coord_lr = args.coord_lr
    color_lr = args.color_lr
    coord_budget = args.coord_budget
    color_budget = args.color_budget
    output_pc_path = args.output_pc_path
    output_visual_pc_path = args.output_visual_pc_path
    pointnum = args.pointnum
    
    if not os.path.exists(output_pc_path):
        os.makedirs(output_pc_path)
        
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, 
                         name=args.wandb_run_name, 
                         reinit=True)

    ori_dataset = load_dataset(args.data_path, args.ori_anno_path, args.pointnum, ("simple_description",), args.use_color)
    tgt_dataset = load_dataset(args.data_path, args.adv_anno_path, args.pointnum, ("simple_description",), args.use_color)
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)
    
    pc_encoder = init_pc_encoder(args).to(device)
    #model.eval()
    
    
    for i, (ori_data_dict, tgt_data_dict) in enumerate(zip(ori_dataloader, tgt_dataloader)):
        ori_pc_ids, ori_pc = ori_data_dict['object_ids'], ori_data_dict['point_clouds']
        tgt_pc_ids, tgt_pc = tgt_data_dict['object_ids'], tgt_data_dict['point_clouds']
        ori_pc = ori_pc.to(device)
        tgt_pc = tgt_pc.to(device)
        with torch.no_grad():
            tgt_feature = pc_encoder(tgt_pc.to(torch.bfloat16))
            if torch.isnan(tgt_feature).any():
                print(f"NaN detected in tgt_feature. Skip this sample.")
                continue
            tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)
            
        delta = torch.zeros_like(ori_pc, requires_grad=True)
        # mask = torch.ones_like(ori_pc, dtype=torch.bool)
        # mask[..., 3:] = 0
        for j in range(args.steps):
            adv_pc = pc_norm(ori_pc + delta)
            adv_feature = pc_encoder(adv_pc.to(torch.bfloat16))
            if torch.isnan(adv_feature).any():
                print(f"NaN detected in adv_feature. Skip this sample.")
                continue
            adv_feature = adv_feature / adv_feature.norm(dim=2, keepdim=True)
            
            embedding_sim = torch.mean(torch.sum(adv_feature * tgt_feature, dim=2))  # computed from normalized features (therefore it is cos sim.)
            embedding_sim.backward()
            
            grad = delta.grad.detach()
            if torch.isnan(grad).any():
                print(f"NaN detected in gradient. Skip this sample.")
                continue
            
            d = torch.zeros_like(delta, requires_grad=False)
            d[..., :3] = torch.clamp(delta[..., :3] + coord_lr * torch.sign(grad[..., :3]), min=-coord_budget, max=coord_budget)
            d[..., 3:] = torch.clamp(delta[..., 3:] + color_lr * torch.sign(grad[..., 3:]), min=-color_budget, max=color_budget)
            delta.data = d
            delta.grad.zero_()

            # Log metrics to wandb
            if args.wandb:
                wandb.log({
                    f"embedding_similarity": embedding_sim.item(),
                    f"max_coord_delta": torch.max(torch.abs(d[..., :3])).item(),
                    f"mean_coord_delta": torch.mean(torch.abs(d[..., :3])).item(),
                    f"max_color_delta": torch.max(torch.abs(d[..., 3:])).item(),
                    f"mean_color_delta": torch.mean(torch.abs(d[..., 3:])).item(),
                })
            print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, \
                  embedding similarity={embedding_sim.item():.5f}, \
                  max coord delta={torch.max(torch.abs(d[..., :3])).item():.3f},   \
                  mean coord delta={torch.mean(torch.abs(d[..., :3])).item():.3f}, \
                  max color delta={torch.max(torch.abs(d[..., 3:])).item():.3f},   \
                  mean color delta={torch.mean(torch.abs(d[..., 3:])).item():.3f}"
                  )
    
        # save adversarial point cloud
        adv_pc = pc_norm(ori_pc + delta)
        for b, (ori_pc_id, tgt_pc_id) in enumerate( zip(ori_pc_ids, tgt_pc_ids) ):
            output_adv_pc_file = os.path.join(output_pc_path, f"{tgt_pc_id}_{pointnum}.npy")
            np.save(output_adv_pc_file, adv_pc[b].cpu().detach().numpy())
            output_visual_pc_file = os.path.join(output_visual_pc_path, f"{ori_pc_id}_{pointnum}", f'adv_coord_{coord_budget}_color_{coord_budget}.txt')
            np.savetxt(output_visual_pc_file, adv_pc[b].cpu().detach().numpy() )
            print(f"Saved visualized adversarial point cloud to {output_visual_pc_file}")
            
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 
    
    parser.add_argument("--coord_lr", type=float, default=0.001)
    parser.add_argument("--color_lr", type=float, default=0.01)
    parser.add_argument("--coord_budget", type=float, default=1.0)
    parser.add_argument("--color_budget", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)

    # data 
    parser.add_argument("--data_path", type=str, default="data/objaverse_data", required=False)
    parser.add_argument("--ori_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_val_200_GT.json", required=False)
    parser.add_argument("--adv_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_adv_200.json", required=False)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color",  action="store_true", default=False)
    parser.add_argument("--output_pc_path", type=str, default="data/adv_pc")
    parser.add_argument("--output_visual_pc_path", type=str, default="data/adv_pc")
    

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)
    
    # logging
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')
    

    args = parser.parse_args()

    main(args)
    
    