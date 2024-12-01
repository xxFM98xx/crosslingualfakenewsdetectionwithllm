# main.py

import os
import argparse
import json
from utils.utils import get_tensorboard_writer
from grid_search import Run
import torch
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--max_len', type=int, default=500)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--language', type=str, default='en')
parser.add_argument('--root_path', type=str)
parser.add_argument('--batchsize', type=int, default=32)#
parser.add_argument('--seed', type=int, default=3759)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=256)  # For LSTM Not used
parser.add_argument('--co_attention_dim', type=int, default=300)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--save_log_dir', type=str, default='./logs')
parser.add_argument('--save_param_dir', type=str, default='./param_model')
parser.add_argument('--param_log_dir', type=str, default='./logs/param')

# Extra parameters
parser.add_argument('--tensorboard_dir', type=str, default='./logs/tensorlog')
parser.add_argument('--bert_path', type=str, default='xlm-roberta-base')
parser.add_argument('--data_type', type=str, default='rationale')
parser.add_argument('--data_name', type=str)
parser.add_argument('--eval_mode', type=bool, default=False)

# Model structure control
parser.add_argument('--expert_interaction_method', type=str, default='cross_attention')
parser.add_argument('--llm_judgment_predictor_weight', type=float, default=1)
parser.add_argument('--rationale_usefulness_evaluator_weight', type=float, default=1)

# Rationales and model variant control
parser.add_argument('--use_rationales', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='RoBERTa_MLP', choices=['RoBERTa_MLP', 'RoBERTa_CNN_MLP','ARG'])
parser.add_argument('--num_filters', type=int, default=100)
parser.add_argument('--mlp_embed_dims', type=str, default='[256]')


# Distill config
parser.add_argument('--kd_loss_weight', type=float, default=1)
parser.add_argument('--teacher_path', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Set random seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {};'.format(
    args.lr, args.model_name, args.batchsize, args.epoch, args.gpu))
print('data_type: {}; data_path: {}; data_name: {};'.format(
    args.data_type, args.root_path, args.data_name))

config = {
    'use_cuda': True,
    'seed': args.seed,
    'batchsize': args.batchsize,
    'max_len': args.max_len,
    'early_stop': args.early_stop,
    'language': args.language,
    'root_path': args.root_path,
    'weight_decay': 5e-5,
    'hidden_size': args.hidden_size, # For LSTM NOT USED
    'model': {
        'mlp': {
            'dims': eval(args.mlp_embed_dims),
            'dropout': args.dropout
        },        'llm_judgment_predictor_weight': args.llm_judgment_predictor_weight,
        'rationale_usefulness_evaluator_weight': args.rationale_usefulness_evaluator_weight,
        'kd_loss_weight': args.kd_loss_weight
    },
    'emb_dim': args.emb_dim,
    'co_attention_dim': args.co_attention_dim,
    'lr': args.lr,
    'epoch': args.epoch,
    'model_name': args.model_name,
    'save_log_dir': args.save_log_dir,
    'save_param_dir': args.save_param_dir,
    'param_log_dir': args.param_log_dir,
    'tensorboard_dir': args.tensorboard_dir,
    'bert_path': args.bert_path,
    'data_type': args.data_type,
    'data_name': args.data_name,
    'eval_mode': args.eval_mode,
    'teacher_path': args.teacher_path,
    'month': 1,
    'use_rationales': args.use_rationales,
    'num_filters': args.num_filters,
}

if __name__ == '__main__':
    writer = get_tensorboard_writer(config)
    print('before in config')
    print(config)
    best_metric = Run(config=config, writer=writer).main()

    save_dir = './logs/log'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, config['data_name'] + '.json')
    with open(save_path, 'w') as file:
        json.dump(best_metric, file, indent=4, ensure_ascii=False)
