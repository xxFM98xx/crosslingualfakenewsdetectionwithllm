# grid_search.py

import logging
import os
import json
import random
import torch
import numpy as np
from models.arg import Trainer as ARGTrainer
from models.baseline import BaselineTrainer

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import itertools

class Run():
    def __init__(self, config, writer):
        self.config = config
        self.writer = writer

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(level=logging.INFO)
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def main(self):
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] + '_' + self.config['data_name'] + '_' + 'param.txt')
        logger = self.getFileLogger(param_log_file)

        # Define hyperparameters to tune
        train_param = {
           'lr': [1e-5, 2e-5],
           'batchsize': [32],
           'dropout': [0.1],
           'num_filters': [100, 200],  # Für CNN
           'mlp_embed_dims': [[256], [256, 128]],  # MLP 
            #'use_rationales': [True, False],  # Whether rationales should be used
            # Additional hyperparameters if necessary needs to be added here
        }
        
        keys, values = zip(*train_param.items())
        best_metric = {'metric': 0}
        best_params = None
        json_result = []

        for v_combination in itertools.product(*values):
            param_combination = dict(zip(keys, v_combination))

            # Set random seed
            setup_seed(self.config['seed'])

            # Update config with current hyperparameters
            self.config['lr'] = param_combination['lr']
            self.config['batchsize'] = param_combination['batchsize']
            self.config['model']['mlp']['dropout'] = param_combination['dropout']
            #self.config['hidden_size'] = param_combination.get('hidden_size', self.config['hidden_size']) # LSTM not used
            self.config['num_filters'] = param_combination.get('num_filters', self.config.get('num_filters', 100))
            self.config['model']['mlp']['dims'] = param_combination.get('mlp_embed_dims', self.config['model']['mlp']['dims'])
            #self.config['use_rationales'] = param_combination.get('use_rationales', self.config['use_rationales'])
            self.config['use_rationales'] = self.config['use_rationales']  # From config

            print('Current params to check: {}'.format(param_combination))
            # Initialisierung des entsprechenden Trainers
            if self.config['model_name'] == 'ARG':
                trainer = ARGTrainer(self.config, self.writer)
            elif self.config['model_name'].startswith('RoBERTa'):
                trainer = BaselineTrainer(self.config, self.writer)
            else:
                raise ValueError('Model name not supported')

            metrics, model_path, train_epochs = trainer.train(logger)
            param_combination['metric'] = metrics
            param_combination['train_epochs'] = train_epochs

            #Save results
            json_result.append(param_combination)

            if metrics['metric'] > best_metric['metric']:
                best_metric = metrics
                best_params = param_combination

            logger.info("Current params: {}".format(param_combination))
            logger.info("Current metric: {}".format(metrics))
            logger.info('==================================================\n\n')

        # Save best hyperparameters and metrics
        json_dir = os.path.join('./logs/json/', self.config['model_name'] + '_' + self.config['data_name'])
        json_path = os.path.join(json_dir, 'month_' + str(self.config['month']) + '.json')
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        with open(json_path, 'w') as file:
            json.dump({'best_params': best_params, 'best_metric': best_metric, 'results': json_result}, file, indent=4, ensure_ascii=False)

        return best_metric