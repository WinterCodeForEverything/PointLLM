#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
#from pointllm.conversation import conv_templates, SeparatorStyle
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
from whitebox.ShapeAttack.HiT_ADV import HiT_ADV
#from whitebox.util.adv_utils import LogitsAdvLoss, CrossEntropyAdvLoss, UntargetedLogitsAdvLoss


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

# def chamfer_distance(pc1, pc2):
#     """
#     Compute the Chamfer Distance between two point clouds.
#     Args:
#         pc1: torch.Tensor of shape (B, N, 6), where B is batch size, N is number of points.
#         pc2: torch.Tensor of shape (B, N, 6), where B is batch size, N is number of points.
#     Returns:
#         torch.Tensor of shape (B,) representing the Chamfer Distance for each batch.
#     """
#     coords1 = pc1[:, :, :3]  # Extract the first 3 dimensions (coordinates)
#     coords2 = pc2[:, :, :3]  # Extract the first 3 dimensions (coordinates)

#     # Compute pairwise distances
#     dist1 = torch.cdist(coords1, coords2, p=2)  # Shape: (B, N, N)
#     dist2 = torch.cdist(coords2, coords1, p=2)  # Shape: (B, N, N)

#     # Compute the minimum distances
#     min_dist1 = dist1.min(dim=2).values  # Shape: (B, N)
#     min_dist2 = dist2.min(dim=2).values  # Shape: (B, N)

#     # Compute the Chamfer Distance
#     chamfer_dist = (min_dist1.mean(dim=1) + min_dist2.mean(dim=1))  # Shape: (B,)

#     return chamfer_dist
   
class FeatureSimilarityLoss(torch.nn.Module):
    def __init__(self, model):
        super(FeatureSimilarityLoss, self).__init__()
        self.model = model
        
    def forward(self, adv_pc, tgt_feature):
        
        if adv_pc.shape[-1] > 6:
            adv_pc = adv_pc[..., :6]
        adv_pc = adv_pc.to(device).to(torch.bfloat16)
        adv_feature = self.model(adv_pc)
        #tgt_feature = self.model(tgt_pc)
        
        # Normalize the features
        adv_feature = adv_feature / adv_feature.norm(dim=2, keepdim=True)
        #print(f"adv_feature: {adv_feature.shape}")
        
        # Compute the embedding similarity
        embedding_sim = torch.mean(torch.sum(adv_feature * tgt_feature, dim=2), dim=1)
        
        return embedding_sim
         
# def adv_loss( adv_pc, tgt_pc, pc_encoder):
#     """
#     Compute the adversarial loss for the point cloud.
#     Args:
#         adv_pc: torch.Tensor of shape (B, N, 6), where B is batch
#             size, N is number of points.
#         tgt_pc: torch.Tensor of shape (B, N, 6), where B is batch
#             size, N is number of points.
#         pc_encoder: Point cloud encoder model.
#     Returns:
#         torch.Tensor representing the adversarial loss.
#     """
#     # Compute the embedding similarity
#     adv_feature = pc_encoder(adv_pc)
#     tgt_feature = pc_encoder(tgt_pc)
    
#     # Normalize the features
#     adv_feature = adv_feature / adv_feature.norm(dim=2, keepdim=True)
#     tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)
    
#     # Compute the embedding similarity
#     embedding_sim = torch.mean(torch.sum(adv_feature * tgt_feature, dim=2))  # computed from normalized features (therefore it is cos sim.)
    
#     return embedding_sim

def main(args):
    

    # alpha = args.alpha
    # epsilon = args.epsilon
    #output_pc_path = args.output_pc_path
    #pointnum = args.pointnum
    # gamma1 = args.gamma1
    
    # if not os.path.exists(output_pc_path):
    #     os.makedirs(output_pc_path)
        
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
    
    #adv_func = CrossEntropyAdvLoss()
    #CW_adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
    adv_func = FeatureSimilarityLoss(pc_encoder)
    HiT_attacker = HiT_ADV( adv_func, attack_lr=args.attack_lr,
                               central_num=args.central_num, total_central_num=args.total_central_num,
                               init_weight=args.init_weight, max_weight=args.max_weight, binary_step=args.binary_step,
                               num_iter=args.num_iter, clip_func=None,
                               cd_weight=args.cd_weight, ker_weight=args.ker_weight,
                               hide_weight=args.hide_weight, curv_loss_knn=args.curv_loss_knn,
                               max_sigm=args.max_sigm, min_sigm=args.min_sigm,
                               budget=args.budget, success_threshold = args.success_threshold,
                               use_wandb = args.wandb )
    
    
    for i, (ori_data_dict, tgt_data_dict) in enumerate(zip(ori_dataloader, tgt_dataloader)):
        _, ori_pc = ori_data_dict['object_ids'], ori_data_dict['point_clouds']
        tgt_pc_id, tgt_pc = tgt_data_dict['object_ids'], tgt_data_dict['point_clouds']
        ori_pc = ori_pc.to(device)
        tgt_pc = tgt_pc.to(device)
        with torch.no_grad():
            tgt_feature = pc_encoder(tgt_pc[..., :6].to(torch.bfloat16))
            if torch.isnan(tgt_feature).any():
                print(f"NaN detected in tgt_feature. Skip this sample.")
                continue
            tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)
            
        results = HiT_attacker.attack(ori_pc, tgt_feature)
            
        # delta = torch.zeros_like(ori_pc, requires_grad=True)
        # mask = torch.ones_like(ori_pc, dtype=torch.bool)
        # mask[..., 3:] = 0
        # for j in range(args.steps):
        #     adv_pc = ori_pc + delta
        #     adv_feature = pc_encoder(adv_pc.to(torch.bfloat16))
        #     if torch.isnan(adv_feature).any():
        #         print(f"NaN detected in adv_feature. Skip this sample.")
        #         continue
        #     adv_feature = adv_feature / adv_feature.norm(dim=2, keepdim=True)
            
        #     embedding_sim = torch.mean(torch.sum(adv_feature * tgt_feature, dim=2))  # computed from normalized features (therefore it is cos sim.)
        #     #embedding_sim.backward()
            
        #     #chamfer_dist = chamfer_distance(adv_pc, tgt_pc).mean()
        #     #print( f"chamfer_dist: {chamfer_dist.mean().item()}")
            
        #     #final_objective = embedding_sim - gamma1 * chamfer_dist
        #     #final_objective.backward()
            
        #     grad = delta.grad.detach()
        #     if torch.isnan(grad).any():
        #         print(f"NaN detected in gradient. Skip this sample.")
        #         continue
        #     d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        #     delta.data = d * mask
        #     delta.grad.zero_()

        #     # Log metrics to wandb
        #     if args.wandb:
        #         wandb.log({
        #             f"embedding_similarity_{i}": embedding_sim.item(),
        #             #f"chamfer_distance_{i}": chamfer_dist.item(),
        #             f"max_delta_{i}": torch.max(torch.abs(d)).item(),
        #             f"mean_delta_{i}": torch.mean(torch.abs(d)).item()
        #         })
        #     print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(d)).item():.3f}, mean delta={torch.mean(torch.abs(d)).item():.3f}")
        #     #chamfer distance = {chamfer_dist.item():.5f},
            
        # # save adversarial point cloud
        # adv_pc = ori_pc + delta
        # for k, pc_id in enumerate(tgt_pc_id):
        #     output_adv_pc_file = os.path.join(output_pc_path, f"{pc_id}_{pointnum}.npy")
        #     np.save(output_adv_pc_file, adv_pc[k].cpu().detach().numpy())
        #     #print(f"Saved adversarial point cloud to {output_adv_pc_file}")
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 
    
    parser.add_argument("--attack_lr", type=float, default=0.01)
    parser.add_argument('--cd_weight', type=float, default=0.0001, help='cd_weight')
    parser.add_argument('--ker_weight', type=float, default=1., help='ker_weight')
    parser.add_argument('--hide_weight', type=float, default=1., help='hide_weight')
    parser.add_argument('--max_sigm', type=float, default=1.2, help='max_sigm')
    parser.add_argument('--min_sigm', type=float, default=0.1, help='min_sigm')
    parser.add_argument('--init_weight', type=float, default=1.0, help='init_weight')
    parser.add_argument('--max_weight', type=float, default=8.0, help='max_weight')
    parser.add_argument('--success_threshold', type=float, default=0.5, help='success_threshold')

    parser.add_argument('--central_num', type=int, default=512)
    parser.add_argument('--total_central_num', type=int, default=1024)
    parser.add_argument('--curv_loss_knn', type=int, default=16, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--budget', type=float, default=0.55,
                        help='0.5 for l2 attack, 0.05 for linf attack')
    
    #CWPerturb_args
    parser.add_argument('--binary_step', type=int, default=10, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=100, metavar='N',
                        help='Number of iterations in each search step')
    
    # parser.add_argument("--epsilon", type=float, default=1.0)
    # parser.add_argument("--gamma1", type=float, default=1.0)
    # parser.add_argument("--steps", type=int, default=300)
    

    # data 
    parser.add_argument("--data_path", type=str, default="data/objaverse_data", required=False)
    parser.add_argument("--ori_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_val_200_GT.json", required=False)
    parser.add_argument("--adv_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_adv_200.json", required=False)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color",  action="store_true", default=False)
    parser.add_argument("--use_normal",  action="store_true", default=False)
    parser.add_argument("--output_pc_path", type=str, default="data/adv_pc")
    

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)
    
    # logging
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')

    args = parser.parse_args()

    main(args)
    
    