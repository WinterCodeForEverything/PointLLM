import argparse
import json
import os
import random
random.seed(0)

import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
#from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm


from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.utils import disable_torch_init
from pointllm.data import ObjectPointCloudDataset
from pointllm.conversation import conv_templates, SeparatorStyle

from pointllm.eval.evaluator import start_evaluation

from whitebox.evaluation.pc_evaluator.pc_evaluator import PointCloudEvaluator

# from evaluation.traditional_evaluator import TraditionalMetricEvaluator
# from evaluation.gpt_evaluator import GPTEvaluator

#from pointllm.eval.traditional_evaluator import TraditionalMetricEvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_LISTS = [
    "What is this object?", #Please describe it in few words, including its color, category and possibly the key characteristics.
    "This is an object of ",
    "Caption this 3D model in detail."
]


def init_model(model_name):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).to(device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv


# class Evaluator():
#     def __init__(self, inputs, output_dir, output_file):
#         self.results = inputs['results']
#         self.inference_prompt = inputs['prompt']
#         self.output_dir = output_dir
#         self.output_file = output_file
#         self.rouge = Rouge()
#         self.response_data = []

#         self.ground_truths = []
#         self.generated_captions = []
        
#         self.traditional_evaluator = TraditionalMetricEvaluator()
#         self.gpt_evaluator = GPTEvaluator()

#         #self.sbert_model = SentenceTransformer('all-mpnet-base-v2')

#         # self.simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
#         # self.simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

#         self.scores = {
#             'bleu-1': [],
#             'bleu-2': [],
#             'bleu-3': [],
#             'bleu-4': [],
#             'rouge-1': [],
#             'rouge-2': [],
#             'rouge-l': [],
#             'meteor': [],
#             'simcse_similarity': [],
#             'gpt_score': [],
#             'success_aa': []
#         }

#     def evaluate_result(self, result):
#         object_id = result['object_id']
#         ori_caption = result['ori_caption']
#         adv_caption = result['adv_caption']
#         tgt_caption = result['tgt_caption']

#         # metrics to evaluate the similarity between adversarial caption and target caption
#         # create a SmoothingFunction object
#         smoothing_function = SmoothingFunction().method1 # * used to deal with non-overlap n-gram

#         # calculate BLEU-1 score with smoothing function
#         bleu_1_score = sentence_bleu([tgt_caption.split()], adv_caption.split(), weights=(1, 0, 0, 0), smoothing_function=smoothing_function)

#         # calculate BLEU-2, BLEU-3, and BLEU-4 scores
#         bleu_2_score = sentence_bleu([tgt_caption.split()], adv_caption.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
#         bleu_3_score = sentence_bleu([tgt_caption.split()], adv_caption.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
#         bleu_4_score = sentence_bleu([tgt_caption.split()], adv_caption.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

#         # calculate ROUGE-L score
#         rouge_scores_l = self.rouge.get_scores(adv_caption, tgt_caption)[0]['rouge-l']
#         rouge_scores_1 = self.rouge.get_scores(adv_caption, tgt_caption)[0]['rouge-1']
#         rouge_scores_2 = self.rouge.get_scores(adv_caption, tgt_caption)[0]['rouge-2']

#         # calculate METEOR score
#         meteor_scores = meteor_score([tgt_caption.split()], adv_caption.split())

#         # # Calculate SBERT similarity
#         # embeddings = self.sbert_model.encode([tgt_caption, adv_caption])
#         # sbert_similarity = util.cos_sim(embeddings[0], embeddings[1])[0][0].item()

#         # calculate SimCSE similarity
#         # Tokenize input texts
#         inputs = self.simcse_tokenizer([tgt_caption, adv_caption], padding=True, truncation=True, return_tensors="pt")

#         # Get the embeddings
#         with torch.no_grad():
#             embeddings = self.simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

#         # Calculate cosine similarity
#         simcse_similarity = 1 - cosine(embeddings[0], embeddings[1]) # * consine actually calculates consine distance, which is 1 - consine similarity
#         simcse_similarity = float(simcse_similarity)
#         # calculate SimCSE similarity between adversarial caption and original caption
#         # Tokenize input texts
#         inputs_ori = self.simcse_tokenizer([ori_caption, adv_caption], padding=True, truncation=True, return_tensors="pt")

#         # Get the embeddings
#         with torch.no_grad():
#             embeddings_ori = self.simcse_model(**inputs_ori, output_hidden_states=True, return_dict=True).pooler_output

#         # Calculate cosine similarity
#         simcse_similarity_ori = 1 - cosine(embeddings_ori[0], embeddings_ori[1])
#         simcse_similarity_ori = float(simcse_similarity_ori)
        
#         scores = {
#             'bleu-1': bleu_1_score * 100,
#             'bleu-2': bleu_2_score * 100,
#             'bleu-3': bleu_3_score * 100,
#             'bleu-4': bleu_4_score * 100,
#             'rouge-l': rouge_scores_l['f'] * 100,
#             'rouge-1': rouge_scores_1['f'] * 100,
#             'rouge-2': rouge_scores_2['f'] * 100,
#             'meteor': meteor_scores * 100,
#             #'sbert_similarity': sbert_similarity * 100,
#             'simcse_similarity': simcse_similarity * 100,
#             'success_aa': float(simcse_similarity > simcse_similarity_ori) * 100
#         }

#         return object_id, ori_caption, adv_caption, tgt_caption, scores 

#     def evaluate(self):
#         print("Starting evaluation...")

#         success_num = 0
#         for result in tqdm(self.results, desc="Evaluating"):  
#             object_id, ori_caption, adv_caption, tgt_caption, scores = self.evaluate_result(result)

#             # save the object_id, model_output, ground_truth, and scores for each result
#             self.response_data.append({
#                 'object_id': object_id,
#                 'ori_caption': ori_caption,
#                 'adv_caption': adv_caption,
#                 'tgt_caption': tgt_caption,
#                 'scores': scores,
#             })

#             # save the scores for overall results
#             for metric, score in scores.items():
#                 self.scores[metric].append(score)
        
#         print("Evaluation finished.")
#         self.save_results()
#         self.print_results()

#     def save_results(self):
#         output_path = os.path.join(self.output_dir, self.output_file)

#         with open(output_path, 'w') as f:
#             results_to_save = {
#                 #'success_rate': f"{np.mean(self.scores['success_aa']):.4f}",
#                 'inference_prompt': self.inference_prompt,
#                 'overall_scores': {metric: f"{np.mean(scores):.4f}" for metric, scores in self.scores.items()},
#                 'results': self.response_data,
#             }
#             json.dump(results_to_save, f, indent=2)
        
#         print(f"Results saved to {output_path}")

#     def print_results(self):
#         print('-' * 80)
#         print("Results:")
#         for metric, scores in self.scores.items():
#             print(f"Average {metric.upper()} Score: {np.mean(scores):.4f}")


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


def generate_outputs( model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
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


def start_generation(model, tokenizer, conv, dataloader, annos, prompt_index, output_dir, output_file):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

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

    input_ids_ = torch.as_tensor(inputs.input_ids).cuda() # * tensor of 1, L

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []

    for batch in tqdm(dataloader):
        point_clouds = batch["point_clouds"].cuda().to(model.dtype) # * tensor of B, N, C(3)
        #print(f"point_clouds shape: {point_clouds.shape}")
        if point_clouds.shape[-1] > 6:
            point_clouds = point_clouds[..., :6].contiguous()
        object_ids = batch["object_ids"] # * list of string 

        batchsize = len(object_ids)

        input_ids = input_ids_.repeat(batchsize, 1) # * tensor of B, L

        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria) # List of str, length is B

        # saving results
        for obj_id, output in zip(object_ids, outputs):
            responses.append({
                "object_id": obj_id,
                "ground_truth": annos[obj_id],
                "model_output": output
            })
    
    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results


# def start_evaluation(results, output_dir, output_file,
#                         parallel=True, num_workers=20):
#     """
#     Args:
#         results: dict or file path to the json file containing the dict
#         output_file: the path the final evaluation results to be saved.
#     """
#     if isinstance(results, str):
#         with open(results, 'r') as fp:
#             results = json.load(fp)

#     evaluator = Evaluator( results, output_dir, output_file ) 
#     evaluator.evaluate()


def start_pc_evaluation(args):
    
    pc_evaluator = PointCloudEvaluator(k_nn=4, device=device)
    ori_dataset = load_dataset(args.ori_pc_path, args.ori_anno_path, args.pointnum, ("simple_description",), args.use_color)
    ori_dataloader = get_dataloader(ori_dataset, args.batch_size, args.shuffle, args.num_workers)
    adv_dataset = load_dataset(args.adv_pc_path, args.adv_anno_path, args.pointnum, ("simple_description",), args.use_color)
    adv_dataloader = get_dataloader(adv_dataset, args.batch_size, args.shuffle, args.num_workers)
    # adv_pc_files = os.listdir(adv_pc_path)    
    # ori_pcs = [np.load(ori_pc_file, allow_pickle=True) ]
    
    results = []
    overall_results = {
            'knn_dist': [],
            'uniform_dist': [],
            'curv_std_dist': []
    }
    for i, (ori_data, adv_data ) in enumerate(zip(ori_dataloader, adv_dataloader)):
        ori_pc = ori_data['point_clouds'].to(device)
        adv_pc = adv_data['point_clouds'].to(device)
        
        ori_xyz = ori_pc[..., :3].contiguous()
        #ori_rgb = ori_pc[:, :, 3:6]
        ori_normal = ori_pc[..., 6:9].contiguous()
        adv_xyz = adv_pc[..., :3].contiguous()
        #adv_rgb = adv_pc[:, :, 3:6]

        # * evaluate the point cloud
        pc_result = pc_evaluator.evaluate(ori_xyz, adv_xyz, ori_normal)  
        for metrics, value in pc_result.items():
            overall_results[metrics].append(value.cpu().numpy())
        pc_result =  {metric: float(value.cpu().numpy()) for metric, value in pc_result.items()}
        pc_result['ori_id'] = ori_data['object_ids']
        pc_result['adv_id'] = adv_data['object_ids']
        results.append(pc_result)

    average_results = {}
    for metrics, values in overall_results.items():
        average_results[metrics] = float(np.mean((values)))
    
    #print(f"Average results: {average_results}")
    print(f"Average knn_dist: {average_results['knn_dist']}")
    print(f"Average uniform_dist: {average_results['uniform_dist']}")
    print(f"Average curv_std_dist: {average_results['curv_std_dist']}")
    
    # * save the results
    save_results = {
        'overall_results': average_results,
        'results': results
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'pc_evaluation_results.json'), 'w') as fp:
        json.dump(save_results, fp, indent=4)

def main(args):
    # * ouptut
    #args.output_dir = os.path.join(args.model_name, "evaluation")
    
    # * output file 
    anno_file = os.path.splitext(os.path.basename(args.adv_anno_path))[0]
    args.output_file = f"{anno_file}_Objaverse_{args.task_type}_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        # * need inferencing
        # * load annotation files
        with open(args.adv_anno_path, 'r') as fp:
            annos = json.load(fp)

        dataset = load_dataset(args.adv_pc_path, args.adv_anno_path, args.pointnum, ("simple_description",), args.use_color)
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
        
        model, tokenizer, conv = init_model(args.model_name)

        # #extract the model output of target pointcloud as anno
        # with open('', 'r') as fp:
        #     tgt_output = json.load(fp)
        # annos = annos['results']
        # annos = {anno["object_id"]: anno['ground_truth'] for anno in annos}
        
        # * convert annos file from [{"object_id": }] to {"object_id": }
        annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}

        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader, annos, args.prompt_index, args.output_dir, args.output_file)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    if args.start_caption_eval:
        evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
        eval_type_mapping = {
            "captioning": "object-captioning",
            "classification": "open-free-form-classification"
        }
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type=eval_type_mapping[args.task_type], model_type=args.gpt_type, parallel=False, num_workers=20)
        
    if args.start_pc_eval:
        start_pc_evaluation( args )
        # pc_evaluator = PointCloudEvaluator(k_nn=4)
        # for result in tqdm(results['results'], desc="Evaluating Point Cloud"):
            
        #     pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 

    # * dataset type
    parser.add_argument("--ori_pc_path", type=str, default="data/objaverse_data", required=True)
    parser.add_argument("--adv_pc_path", type=str, default="data/objaverse_data", required=True)
    parser.add_argument("--ori_anno_path", type=str, default="data/anno_data/ori_annos.json", required=True)
    parser.add_argument("--adv_anno_path", type=str, default="data/anno_data/adv_annos.json", required=True)
    parser.add_argument("--output_dir", type=str, default="", required=True)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color",  action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_caption_eval", action="store_true", default=False)
    parser.add_argument("--start_pc_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4o-mini", choices=[ "gpt-4.1-mini", "gpt-4o-mini"], help="Type of the model used to evaluate.")
    parser.add_argument("--task_type", type=str, default="captioning", choices=["captioning", "classification"], help="Type of the task to evaluate.")

    args = parser.parse_args()

    # * check prompt index
    # * * classification: 0, 1 and captioning: 2. Raise Warning otherwise.
    if args.task_type == "classification":
        if args.prompt_index != 0 and args.prompt_index != 1:
            print("[Warning] For classification task, prompt_index should be 0 or 1.")
    elif args.task_type == "captioning":
        if args.prompt_index != 2:
            print("[Warning] For captioning task, prompt_index should be 2.")
    else:
        raise NotImplementedError

    main(args)  

    
    # results = './result/noise_size_5_result/PointLLM_brief_description_mrg_200_Objaverse_classification_prompt0.json'
    # output_dir = './result/noise_size_5_result/MF_pp'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # output_file = 'traditional_metrics.json'
    # start_evaluation(results, output_dir, output_file)