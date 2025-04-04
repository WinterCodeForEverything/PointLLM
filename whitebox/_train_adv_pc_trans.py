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

def main(args):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = args.alpha
    epsilon = args.epsilon
    output_pc_path = args.output_pc_path
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
        _, ori_pc = ori_data_dict['object_ids'], ori_data_dict['point_clouds']
        tgt_pc_id, tgt_pc = tgt_data_dict['object_ids'], tgt_data_dict['point_clouds']
        ori_pc = ori_pc.to(device)
        tgt_pc = tgt_pc.to(device)
        with torch.no_grad():
            tgt_feature = pc_encoder(tgt_pc.to(torch.bfloat16))
            #print(tgt_feature.shape)
            if torch.isnan(tgt_feature).any():
                print(f"NaN detected in tgt_feature. Skip this sample.")
                continue
            tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)
            
        delta = torch.zeros_like(ori_pc, requires_grad=True)
        for j in range(args.steps):
            adv_pc = ori_pc + delta
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
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = d
            delta.grad.zero_()


            # Log metrics to wandb
            if args.wandb:
                wandb.log({
                    f"embedding_similarity_{i}": embedding_sim.item(),
                    f"max_delta_{i}": torch.max(torch.abs(d)).item(),
                    f"mean_delta_{i}": torch.mean(torch.abs(d)).item()
                })
            print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(d)).item():.3f}, mean delta={torch.mean(torch.abs(d)).item():.3f}")
    
        # save adversarial point cloud
        adv_pc = ori_pc + delta
        for k, pc_id in enumerate(tgt_pc_id):
            output_adv_pc_file = os.path.join(output_pc_path, f"{pc_id}_{pointnum}.npy")
            np.save(output_adv_pc_file, adv_pc[k].cpu().detach().numpy())
            #print(f"Saved adversarial point cloud to {output_adv_pc_file}")
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 
    
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epsilon", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)

    # data 
    parser.add_argument("--data_path", type=str, default="data/objaverse_data", required=False)
    parser.add_argument("--ori_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_val_200_GT.json", required=False)
    parser.add_argument("--adv_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_adv_200.json", required=False)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color",  action="store_true", default=False)
    parser.add_argument("--output_pc_path", type=str, default="data/adv_pc")
    

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
    
    