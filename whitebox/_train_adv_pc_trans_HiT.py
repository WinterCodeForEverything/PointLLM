#from pointllm.model.pointllm import PointLLMLlamaForCausalLM, PointLLMLlamaModel 
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from pointllm.utils import disable_torch_init
from pointllm.model import *
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
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).to(device)
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


def main(args):
    
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
    
    # model, tokenizer, conv  = init_model(args.model_name)
    # model.eval()
    # pc_encoder = model.get_pc_encoder()
    pc_encoder = init_pc_encoder(args)
    
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
    
    # # prepere for evaluation
    # batch_size = args.batch_size
    # point_backbone_config = model.get_model().point_backbone_config
    # point_token_len = point_backbone_config['point_token_len']
    # default_point_patch_token = point_backbone_config['default_point_patch_token']
    # default_point_start_token = point_backbone_config['default_point_start_token']
    # default_point_end_token = point_backbone_config['default_point_end_token']
    # mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    
    # if mm_use_point_start_end:
    #     default_point_prompt = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n'
    # else:
    #     default_point_prompt = default_point_patch_token * point_token_len + '\n'
    
    
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # qs = default_point_prompt + PROMPT_LISTS[args.prompt_index]
    
    # conv.append_message(conv.roles[0], qs)
    # conv.append_message(conv.roles[1], None)

    # prompt = conv.get_prompt()
    # inputs = tokenizer([prompt])
    # prefix_input_ids = torch.as_tensor( inputs.input_ids ).to(device) # * tensor of 1, L
    # stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, prefix_input_ids)
    # prefix_input_ids = prefix_input_ids.repeat(batch_size, 1)
    
    
    
    # gptEvaluator = GPTEvaluator( args.gpt_type )
    # traditianalEvaluator = TraditionalMetricEvaluator()
    # evaluate_results = []
    
    # all_scores = {
    #     'bleu-1': [],
    #     'bleu-2': [],
    #     'bleu-3': [],
    #     'bleu-4': [],
    #     'rouge-1': [],
    #     'rouge-2': [],
    #     'rouge-l': [],
    #     'meteor': [],
    #     'simcse_similarity': [],
    #     'gpt_score': [],
    #     'success_aa': [],
    # }
    for i, (ori_data_dict, tgt_data_dict) in enumerate(zip(ori_dataloader, tgt_dataloader)):
        ori_pc_ids, ori_pc = ori_data_dict['object_ids'], ori_data_dict['point_clouds']
        tgt_pc_ids, tgt_pc = tgt_data_dict['object_ids'], tgt_data_dict['point_clouds']
        ori_pc = ori_pc.to(device)
        tgt_pc = tgt_pc.to(device)
        with torch.no_grad():
            tgt_feature = pc_encoder(tgt_pc[..., :6].to(torch.bfloat16))
            if torch.isnan(tgt_feature).any():
                print(f"NaN detected in tgt_feature. Skip this sample.")
                continue
            tgt_feature = tgt_feature / tgt_feature.norm(dim=2, keepdim=True)
            
        adv_pc, _ = HiT_attacker.attack(ori_pc, tgt_feature)
        #adv_pc = torch.from_numpy(adv_pc).to(device)
        
        # save adversarial point cloud
        for b, (ori_pc_id, tgt_pc_id) in enumerate( zip(ori_pc_ids, tgt_pc_ids) ):
            output_adv_pc_file = os.path.join(output_pc_path, f"{tgt_pc_id}_{pointnum}.npy")
            np.save(output_adv_pc_file, adv_pc[b] )
            output_visual_pc_file = os.path.join(output_visual_pc_path, f"{ori_pc_id}_{pointnum}", f'adv_HiT_{args.success_threshold}.txt')
            np.savetxt(output_visual_pc_file, adv_pc[b] )
            print(f"Saved visualized adversarial point cloud to {output_visual_pc_file}")
        
    #     #start evalution
    #     adv_captions = generate_outputs(model, tokenizer, prefix_input_ids, adv_pc, stopping_criteria)
    #     tgt_captions = generate_outputs(model, tokenizer, prefix_input_ids, tgt_pc, stopping_criteria)
    #     for b, (ori_pc_id, tgt_pc_id) in enumerate(zip(ori_pc_ids, tgt_pc_ids) ):
    #         adv_caption = adv_captions[b]
    #         tgt_caption = tgt_captions[b]
    #         gpt_result = gptEvaluator.evaluate(adv_caption, tgt_caption)
    #         traditianal_result = traditianalEvaluator.evaluate( adv_caption, tgt_caption )
            
            
    #         evaluate_result ={
    #             'ori_pc_id': ori_pc_id,
    #             'tgt_pc_id': tgt_pc_id,
    #             'adv_caption': adv_caption,
    #             'tgt_caption': tgt_caption,
    #             'gpt_result': gpt_result,
    #             'traditianal_result': traditianal_result,
    #         }
    #         evaluate_results.append(evaluate_result)
    #         print(evaluate_result)
            
    #         for metric, score in traditianal_result.items():
    #             all_scores[metric].append(score)
    #         all_scores['gpt_score'].append(gpt_result['gpt_score'])
    #         all_scores['success_aa'].append( gpt_result['gpt_score'] >= 50 )

    # avg_scores =  {}
    # for metric, scores in avg_scores.items():    
    #     avg_scores[metric] = np.mean(scores)
        

        
    # # print average scores
    # print('-' * 80)
    # print("Results:")
    # for metric, score in avg_scores.items():
    #     print(f"Average {metric.upper()} Score: {score:.4f}")
        
    # # save the evaluation results
    # save_results ={
    #     'prompt': PROMPT_LISTS[args.prompt_index],
    #     'avg_scores': avg_scores,
    #     'results': evaluate_results
    # }
    # output_eval_file = os.path.join(output_evaluation_path, f"evaluate_results.json")
    # if not os.path.exists(output_evaluation_path):
    #     os.makedirs(output_evaluation_path)
    # with open(output_eval_file, 'w') as f:
    #     json.dump(save_results, f, indent=4)
    # print(f"Saved evaluation results to {output_eval_file}")

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
    parser.add_argument("--output_visual_pc_path", type=str, default="data/adv_pc")
    

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
    
    