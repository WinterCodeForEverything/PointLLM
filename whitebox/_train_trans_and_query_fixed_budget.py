
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.data import ObjectPointCloudDataset
from tqdm import tqdm
from transformers import AutoTokenizer
# from pointllm.eval.evaluator import start_evaluation
import os
import json
import wandb

PROMPT_LISTS = [
    "What is this object? Please describe it in few words, including its color, category and possibly the key characteristics.",
    "This is an object of ",
    "Caption this 3D model in detail."
]

device = "cuda" if torch.cuda.is_available() else "cpu"

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    
    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()
    
    txt_encoder = model.get_txt_encoder()

    return model, tokenizer, conv, txt_encoder


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


def generate_caption(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria]) # * B, L'
        

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs

def get_text_embedding(captions, tokenizer, txt_encoder, max_length=None):
    """
    Generate text embeddings for the given captions using the tokenizer and text encoder.

    Args:
        captions (list of str): List of text captions to encode.
        tokenizer: Tokenizer to process the text captions.
        txt_encoder: Text encoder model to generate embeddings.
        max_length (int, optional): Maximum length for tokenized input. Defaults to None.

    Returns:
        torch.Tensor: Normalized embeddings for the total token representing the complete text.
    """
    inputs = tokenizer(captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # Get the output embeddings from the text encoder
    outputs = txt_encoder(input_ids)
    text_embedding = torch.mean(outputs, dim=1)  # Average pooling over the sequence length

    # Normalize the embedding
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.detach()
    return text_embedding

def main(args):

    batch_size = args.batch_size
    alpha = args.alpha
    epsilon = args.epsilon
    #sigma = args.sigma
    output_pc_path = args.output_pc_path
    output_showing_pc_path = args.output_showing_pc_path
    #pointnum = args.pointnum
    prompt_index = args.prompt_index
    num_query = args.num_query
    num_sub_query = args.num_sub_query
    max_length = args.max_caption_length
    
    assert num_query % num_sub_query == 0, "num_query should be divisible by num_sub_query"
    
    output_pc_path = f'{output_pc_path}__epsilon_{epsilon}'
    if not os.path.exists(output_pc_path):
        os.makedirs(output_pc_path)
        
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, 
                         name=args.wandb_run_name, 
                         reinit=True)
    
    
    # initialize models
    model, tokenizer, conv, txt_encoder = init_model(args)
    model = model.to(device)
    txt_encoder = txt_encoder.to(device)
    
    # initialize input text tokens
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    input_ids_ = torch.as_tensor(inputs.input_ids).to(device) # * tensor of 1, L
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)
        
    # load adv and tgt text and extract text feature
    adv_captions, tgt_captions = [], []
    with open(args.mrg_output_file, "r") as json_file:
        list_tgt_cap_dict = json.load(json_file)
        for result in list_tgt_cap_dict['results']:
            adv_captions.append(result['adv_caption'])
            tgt_captions.append(result['tgt_caption'])

    # Determine the maximum length between adv_captions and tgt_captions
    #max_length = max(len(tokenizer.tokenize(caption)) for caption in adv_captions + tgt_captions)

    # Tokenize adv_captions and tgt_captions with padding to the same length
    adv_text_feature = get_text_embedding(adv_captions, tokenizer, txt_encoder, max_length)
    tgt_text_feature = get_text_embedding(tgt_captions, tokenizer, txt_encoder, max_length)
    
    init_embedding_similarity    = torch.sum(adv_text_feature.float() * tgt_text_feature.float(), dim=-1).cpu().numpy()
    query_attack_results  = torch.sum(adv_text_feature.float() * tgt_text_feature.float(), dim=-1).cpu().numpy()
    
    # print(f"adv_text_feature.shape: {adv_text_feature.shape}")
    # print(f"tgt_text_feature.shape: {tgt_text_feature.shape}")
    
    # the adverearial dataset from MF-pp as original dataset for MF-tt
    clean_dataset = load_dataset(args.ori_data_path, args.ori_anno_path, args.pointnum, ("simple_description",), args.use_color)
    ori_dataset = load_dataset(args.adv_data_path, args.adv_anno_path, args.pointnum, ("simple_description",), args.use_color)
    #tgt_dataset = load_dataset(args.ori_data_path, args.tgt_anno_path, args.pointnum, ("simple_description",), args.use_color)
    clean_dataloader = get_dataloader(clean_dataset, args.batch_size, args.shuffle, args.num_workers)
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    #tgt_dataloader = get_dataloader(tgt_dataset, args.batch_size, args.shuffle, args.num_workers)
    
    for i, (ori_data_dict, clean_data_dict) in enumerate(zip( ori_dataloader, clean_dataloader)):
        clean_object_ids, clean_pc = clean_data_dict['object_ids'], clean_data_dict['point_clouds']
        _, ori_pc = ori_data_dict['object_ids'], ori_data_dict['point_clouds']
        clean_pc = clean_pc.to(device)
        ori_pc = ori_pc.to(device)
        
        upbound = batch_size * (i+1) if batch_size * (i+1) < len(adv_text_feature) else len(adv_text_feature)
        adv_text_features_i = adv_text_feature[batch_size * (i): upbound]    
        tgt_text_features_i = tgt_text_feature[batch_size * (i): upbound]
        # print(f"adv_text_features_i.shape: {adv_text_features_i.shape}")
        # print(f"tgt_text_features_i.shape: {tgt_text_features_i.shape}")
        
        # ------------------- random gradient-free method
        print("init delta with diff(adv-clean)")
        delta = (ori_pc - clean_pc).clone().detach()
        torch.cuda.empty_cache()
        
        
        
        for step_idx in range(args.steps):
            print(f"{i}-th points / {step_idx}-th step")
            
            # step 1. obtain purturbed points
            with torch.no_grad():
                if step_idx == 0:
                    pc_repeat = clean_pc.repeat(num_query, 1, 1)
                    
                else:
                    pc_repeat   = adv_pc_in_current_step.repeat(num_query, 1, 1) 
                    
                    input_ids = input_ids_.repeat(batch_size, 1) # * tensor of B, L
                    generate_captions_in_current_step = generate_caption(model, tokenizer, input_ids, adv_pc_in_current_step, stopping_criteria)
                    adv_text_feature_in_current_step = get_text_embedding(generate_captions_in_current_step, tokenizer, txt_encoder, max_length)
                    # adv_text_feature_in_current_step = generate_output_embedding(model,  input_ids, adv_pc_in_current_step )
                    # adv_text_feature_in_current_step = adv_text_feature_in_current_step / adv_text_feature_in_current_step.norm(dim=-1, keepdim=True)
                    # adv_text_feature_in_current_step = adv_text_feature_in_current_step.detach()

                    adv_text_features_i = adv_text_feature_in_current_step
                    

            query_noise            = torch.randn_like(pc_repeat).sign() # Rademacher noise
            perturbed_pc_repeat = torch.clamp(pc_repeat + (alpha * query_noise), min=-epsilon, max=epsilon) 
            perturbed_pc_repeat = perturbed_pc_repeat.to(device).to(model.dtype)
            
            
            perturb_text_features = []
            for j in range(num_query//num_sub_query):
                perturbed_pc = perturbed_pc_repeat[j*num_sub_query:(j+1)*num_sub_query]
                input_ids = input_ids_.repeat(batch_size*num_sub_query, 1)
                with torch.no_grad():
                    #text_embedding_of_perturbed_pc = generate_output_embedding(model, txt_encoder, input_ids, perturbed_pc, stopping_criteria)
                    perturbed_sub_text = generate_caption(model, tokenizer, input_ids, perturbed_pc, stopping_criteria)
                    perturb_sub_text_features = get_text_embedding(perturbed_sub_text, tokenizer, txt_encoder, max_length)
                    # perturb_sub_text_features = generate_output_embedding(model,  input_ids, perturbed_pc )
                    # perturb_sub_text_features = perturb_sub_text_features / perturb_sub_text_features.norm(dim=-1, keepdim=True)
                    # perturb_sub_text_features = perturb_sub_text_features.detach()
                    perturb_text_features.append(perturb_sub_text_features)
                    
            perturb_text_features = torch.cat(perturb_text_features)
            #print( 'perturb_text_features:', perturb_text_features.shape )
            
            # step 2. estimate grad
            coefficient     = torch.sum((perturb_text_features - adv_text_features_i) * tgt_text_features_i, dim=-1)
            coefficient     = coefficient.reshape(num_query, batch_size, 1, 1)
            query_noise     = query_noise.reshape(num_query, batch_size, -1, 6)
            pseudo_gradient = coefficient * query_noise     #/ alpha 
            pseudo_gradient = pseudo_gradient.mean(0) 
            
            #adv_text_feature_in_current_step = generate_output_embedding(model,  input_ids, adv_pc_in_current_step )
        
            # step 3. log metrics
            delta_data = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
            delta.data = delta_data
            adv_pc_in_current_step = torch.clamp(clean_pc+delta, min=-epsilon, max=epsilon).to(model.dtype)

            print(f"{i}-th points // {step_idx}-th step // max  delta", torch.max(torch.abs(delta)).item())
            print(f"{i}-th points // {step_idx}-th step // mean delta", torch.mean(torch.abs(delta)).item())
            
            current_embedding_similarity = torch.sum(adv_text_features_i.float() * tgt_text_features_i.float()).cpu().numpy()
            print(f"{i}-th points // {step_idx}-th step // embedding similarity", current_embedding_similarity)
            if current_embedding_similarity > query_attack_results[i]:
                    query_attack_results[i] = current_embedding_similarity
                    
            torch.cuda.empty_cache()
        

        if args.wandb:
            wandb.log({
                f"init_embedding_similarity": np.mean(init_embedding_similarity[:(i+1)]),
                f"embedding_similarity_after_query_attack": np.mean(query_attack_results[:(i+1)]),
            })
        
        
        
        adv_pc = torch.clamp(clean_pc+delta, min=-epsilon, max=epsilon)
        adv_pc = ori_pc + delta
        for k, pc_id in enumerate(clean_object_ids):
            # save showing point cloud
            output_showing_pc_file_path = os.path.join(output_showing_pc_path, f'{pc_id}_{args.pointnum}' )
            if not os.path.exists(output_showing_pc_file_path):
                print(f"No Exixted directory: {output_showing_pc_file_path}")
                continue
            output_showing_adv_pc_file = os.path.join(output_showing_pc_file_path, f"pp_tt_adv_{epsilon}.txt")
            np.savetxt(output_showing_adv_pc_file, adv_pc[k].cpu().detach().numpy())
            
            # save adversarial point cloud for evaluation
            output_adv_pc_file = os.path.join(output_pc_path, f"{pc_id}_{args.pointnum}.npy")
            np.save(output_adv_pc_file, adv_pc[k].cpu().detach().numpy())

    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 
    
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.1)
    #parser.add_argument("--sigma", type=int, default=0.1)
    parser.add_argument("--num_query", type=int, default=10)
    parser.add_argument("--num_sub_query", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prompt_index", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--max_caption_length", type=int, default=256)

    # data 
    parser.add_argument("--ori_data_path", type=str, default="data/objaverse_data", required=False)
    parser.add_argument("--adv_data_path", type=str, default="data/Objaverse_adv_npy", required=False)
    parser.add_argument("--ori_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_val_200_GT.json", required=False)
    parser.add_argument("--adv_anno_path", type=str, default="data/anno_data/PointLLM_brief_description_adv_200.json", required=False)
    parser.add_argument("--mrg_output_file", type=str, default="", required=True)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color",  action="store_true", default=False)
    parser.add_argument("--output_pc_path", type=str, default="data/adv_pc")
    parser.add_argument("--output_showing_pc_path", type=str, default="data/adv_pc")

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