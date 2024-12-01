# models/baseline.py

import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from utils.dataloader import get_dataloader_baseline
from utils.utils import data2gpu
from utils.utils import Averager, Recorder
import time
import tqdm
from utils.utils import metrics
from .layers import MLP
import os

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.use_rationales = config.get('use_rationales', True)

        # XLM-RoBERTa Encoder for the content
        self.bert_content = XLMRobertaModel.from_pretrained(config['bert_path'])

        if self.use_rationales:
            # XLM-RoBERTa Encoder for the td_rationale (td = linguistic style)
            self.bert_td_rationale = XLMRobertaModel.from_pretrained(config['bert_path'])
            # XLM-RoBERTa Encoder for the cs_rationale (cs = commonsense)
            self.bert_cs_rationale = XLMRobertaModel.from_pretrained(config['bert_path'])

        # english: freeze all parameters except the last layer 
        self.freeze_parameters()

    def freeze_parameters(self):
        """        
        This method is used to freeze all parameters of the model except the last layer.
        """
        # Freeze all parameters
        for param in self.bert_content.parameters():
            param.requires_grad = False
        if self.use_rationales:
            for param in self.bert_td_rationale.parameters():
                param.requires_grad = False
            for param in self.bert_cs_rationale.parameters():
                param.requires_grad = False

        # Fine tuning the last encoder layer
        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
        if self.use_rationales:
            for name, param in self.bert_td_rationale.named_parameters():
                if name.startswith("encoder.layer.11"):
                    param.requires_grad = True
            for name, param in self.bert_cs_rationale.named_parameters():
                if name.startswith("encoder.layer.11"):
                    param.requires_grad = True
                    
class RoBERTa_MLP(BaseModel):
    def __init__(self, config):
        super(RoBERTa_MLP, self).__init__(config)

        # Input dimension for the MLP
        if self.use_rationales:
            mlp_input_dim = config['emb_dim'] * 3  # Content + td_rationale + cs_rationale
        else:
            mlp_input_dim = config['emb_dim']  # Only content

        # MLP for classification
        self.mlp = MLP(
            input_dim=mlp_input_dim,
            embed_dims=config['model']['mlp']['dims'],
            dropout=config['model']['mlp']['dropout']
        )

    def forward(self, content_ids, content_masks, td_rationale_ids=None, td_rationale_masks=None, cs_rationale_ids=None, cs_rationale_masks=None):
        # BERT outputs for the content
        content_outputs = self.bert_content(content_ids, attention_mask=content_masks)
        content_cls = content_outputs.last_hidden_state[:, 0, :]

        if self.use_rationales:
            # BERT outputs for the td_rationale
            td_rationale_outputs = self.bert_td_rationale(td_rationale_ids, attention_mask=td_rationale_masks)
            td_rationale_cls = td_rationale_outputs.last_hidden_state[:, 0, :]

            # BERT outputs for the cs_rationale
            cs_rationale_outputs = self.bert_cs_rationale(cs_rationale_ids, attention_mask=cs_rationale_masks)
            cs_rationale_cls = cs_rationale_outputs.last_hidden_state[:, 0, :]

            # Concatenation of the features
            combined_features = torch.cat((content_cls, td_rationale_cls, cs_rationale_cls), dim=1)
        else:
            combined_features = content_cls  # Only content features

        # Klassifikation
        logits = self.mlp(combined_features)
        return logits

# NOT IN USE
class RoBERTa_LSTM_MLP(BaseModel):
    def __init__(self, config):
        super(RoBERTa_LSTM_MLP, self).__init__(config)

        # LSTM for the content
        self.lstm_content = nn.LSTM(
            input_size=config['emb_dim'],
            hidden_size=config['hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        if self.use_rationales:
            # LSTM for the td_rationale
            self.lstm_td_rationale = nn.LSTM(
                input_size=config['emb_dim'],
                hidden_size=config['hidden_size'],
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

            # LSTM for the cs_rationale
            self.lstm_cs_rationale = nn.LSTM(
                input_size=config['emb_dim'],
                hidden_size=config['hidden_size'],
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

        # Input dimension for the MLP
        if self.use_rationales:
            mlp_input_dim = config['emb_dim']* 6  # Content + td_rationale + cs_rationale (jeweils bidirektional)
        else:
            mlp_input_dim = config['emb_dim']* 2  # Nur Content (bidirektional)

        # MLP for classification
        self.mlp = MLP(
            input_dim=mlp_input_dim,
            embed_dims=config['model']['mlp']['dims'],
            dropout=config['model']['mlp']['dropout']
        )

    def forward(self, content_ids, content_masks, td_rationale_ids=None, td_rationale_masks=None, cs_rationale_ids=None, cs_rationale_masks=None):
        # BERT outputs for the content
        content_outputs = self.bert_content(content_ids, attention_mask=content_masks)
        content_sequence = content_outputs.last_hidden_state  # Shape: (batch_size, seq_len, emb_dim)

        # LSTM over the content
        content_lstm_out, _ = self.lstm_content(content_sequence)
        content_pooled = torch.mean(content_lstm_out, dim=1)

        if self.use_rationales:
            # BERT outputs for the td_rationale
            td_rationale_outputs = self.bert_td_rationale(td_rationale_ids, attention_mask=td_rationale_masks)
            td_rationale_sequence = td_rationale_outputs.last_hidden_state

            # LSTM über die td_rationale
            td_rationale_lstm_out, _ = self.lstm_td_rationale(td_rationale_sequence)
            td_rationale_pooled = torch.mean(td_rationale_lstm_out, dim=1)

            # BERT outputs for the cs_rationale
            cs_rationale_outputs = self.bert_cs_rationale(cs_rationale_ids, attention_mask=cs_rationale_masks)
            cs_rationale_sequence = cs_rationale_outputs.last_hidden_state

            # LSTM über die cs_rationale
            cs_rationale_lstm_out, _ = self.lstm_cs_rationale(cs_rationale_sequence)
            cs_rationale_pooled = torch.mean(cs_rationale_lstm_out, dim=1)

            # Concatenation of the features
            combined_features = torch.cat((content_pooled, td_rationale_pooled, cs_rationale_pooled), dim=1)
        else:
            combined_features = content_pooled  # Only content features

        # Klassifikation
        logits = self.mlp(combined_features)
        return logits
    
class RoBERTa_CNN_MLP(BaseModel):
    def __init__(self, config):
        super(RoBERTa_CNN_MLP, self).__init__(config)

        # CNN for the content
        self.conv_content = nn.Conv1d(
            in_channels=config['emb_dim'],
            out_channels=config['num_filters'],
            kernel_size=3,
            padding=1
        )

        if self.use_rationales:
            # CNN for the td_rationale
            self.conv_td_rationale = nn.Conv1d(
                in_channels=config['emb_dim'],
                out_channels=config['num_filters'],
                kernel_size=3,
                padding=1
            )

            # CNN for the cs_rationale
            self.conv_cs_rationale = nn.Conv1d(
                in_channels=config['emb_dim'],
                out_channels=config['num_filters'],
                kernel_size=3,
                padding=1
            )

        # Input dimension for the MLP
        if self.use_rationales:
            mlp_input_dim = config['num_filters'] * 3  # Content + td_rationale + cs_rationale
        else:
            mlp_input_dim = config['num_filters']  # Only content

        # MLP for classification
        self.mlp = MLP(
            input_dim=mlp_input_dim,
            embed_dims=config['model']['mlp']['dims'],
            dropout=config['model']['mlp']['dropout']
        )

    def forward(self, content_ids, content_masks, td_rationale_ids=None, td_rationale_masks=None, cs_rationale_ids=None, cs_rationale_masks=None):
        # BERT outputs for the content
        content_outputs = self.bert_content(content_ids, attention_mask=content_masks)
        content_sequence = content_outputs.last_hidden_state.permute(0, 2, 1)  # (batch_size, emb_dim, seq_len)

        # CNN über den Inhalt
        content_conv = self.conv_content(content_sequence)
        content_conv = torch.relu(content_conv)
        content_pooled = torch.max(content_conv, dim=2)[0]

        if self.use_rationales:
            # BERT outputs for the td_rationale
            td_rationale_outputs = self.bert_td_rationale(td_rationale_ids, attention_mask=td_rationale_masks)
            td_rationale_sequence = td_rationale_outputs.last_hidden_state.permute(0, 2, 1)

            # CNN über die td_rationale
            td_rationale_conv = self.conv_td_rationale(td_rationale_sequence)
            td_rationale_conv = torch.relu(td_rationale_conv)
            td_rationale_pooled = torch.max(td_rationale_conv, dim=2)[0]

            # BERT outputs for the cs_rationale
            cs_rationale_outputs = self.bert_cs_rationale(cs_rationale_ids, attention_mask=cs_rationale_masks)
            cs_rationale_sequence = cs_rationale_outputs.last_hidden_state.permute(0, 2, 1)

            # CNN über die cs_rationale
            cs_rationale_conv = self.conv_cs_rationale(cs_rationale_sequence)
            cs_rationale_conv = torch.relu(cs_rationale_conv)
            cs_rationale_pooled = torch.max(cs_rationale_conv, dim=2)[0]

            # Concatenation of the features
            combined_features = torch.cat((content_pooled, td_rationale_pooled, cs_rationale_pooled), dim=1)
        else:
            combined_features = content_pooled  # Only content features

        # Klassifikation
        #Classification
        logits = self.mlp(combined_features)
        return logits

class BaselineTrainer():
    def __init__(self, config, writer):
        self.config = config
        self.writer = writer
        self.save_path = os.path.join(
            self.config['save_param_dir'],
            self.config['model_name'] + '_' + self.config['data_name'],
            str(self.config['month'])
        )
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def initialize_model(self):
        if self.config['model_name'] == 'RoBERTa_MLP':
            model = RoBERTa_MLP(self.config)
        #elif self.config['model_name'] == 'RoBERTa_LSTM_MLP': Not in use
        #    model = RoBERTa_LSTM_MLP(self.config)
        elif self.config['model_name'] == 'RoBERTa_CNN_MLP':
            model = RoBERTa_CNN_MLP(self.config)
        else:
            model = BaseModel(self.config)
        return model

    def train(self, logger=None):
        st_tm = time.time()
        writer = self.writer

        if logger:
            logger.info('Start training...')

        print('\n\n')
        print('==================== Start Training ====================')

        self.model = self.initialize_model()

        if self.config['use_cuda']:
            self.model = self.model.cuda()

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        recorder = Recorder(self.config['early_stop'])

        # Load data
        train_path = os.path.join(self.config['root_path'], 'train.json')
        train_loader = get_dataloader_baseline(
            train_path,
            self.config['max_len'],
            self.config['batchsize'],
            shuffle=True,
            bert_path=self.config['bert_path'],
            data_type=self.config['data_type'],
            language=self.config['language']
        )

        val_path = os.path.join(self.config['root_path'], 'val.json')
        val_loader = get_dataloader_baseline(
            val_path,
            self.config['max_len'],
            self.config['batchsize'],
            shuffle=False,
            bert_path=self.config['bert_path'],
            data_type=self.config['data_type'],
            language=self.config['language']
        )

        test_path = os.path.join(self.config['root_path'], 'test.json')
        test_loader = get_dataloader_baseline(
            test_path,
            self.config['max_len'],
            self.config['batchsize'],
            shuffle=False,
            bert_path=self.config['bert_path'],
            data_type=self.config['data_type'],
            language=self.config['language']
        )

        ed_tm = time.time()
        print('Time cost in model and data loading: {}s'.format(ed_tm - st_tm))

        # Training loop
        for epoch in range(self.config['epoch']):
            print('---------- Epoch {} ----------'.format(epoch))
            self.model.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'], data_type=self.config['data_type'], model_name=self.config['model_name'])
                labels = batch_data['label'].float().unsqueeze(1)

                logits = self.model(
                    batch_data['content'],
                    batch_data['content_masks'],
                    batch_data.get('td_rationale'),
                    batch_data.get('td_rationale_masks'),
                    batch_data.get('cs_rationale'),
                    batch_data.get('cs_rationale_masks')
                )

                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())

            print('----- In validation progress... -----')
            results, val_aux_info = self.test(val_loader)
            mark = recorder.add(results)
            print()

            # TensorBoard logging
            writer.add_scalar('month_' + str(self.config['month']) + '/train_loss', avg_loss.item(), global_step=epoch)
            writer.add_scalars('month_' + str(self.config['month']) + '/val', results, global_step=epoch)

            # Logger
            if logger:
                logger.info('---------- Epoch {} ----------'.format(epoch))
                logger.info('Train loss: {}'.format(avg_loss.item()))
                logger.info('Validation result: {}'.format(results))
                logger.info('\n')

            # Early stopping and model saving
            if mark == 'save':
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'parameter.pkl'))
            if mark == 'esc':
                break

        # Load the best model after training and make predictions
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter.pkl')))
        test_results, label, pred, id_list, ae, accuracy = self.predict(test_loader)

        writer.add_scalars('month_' + str(self.config['month']) + '/test', test_results)

        if logger:
            logger.info("Start testing...")
            logger.info("Test score: {}.".format(test_results))
            logger.info("Learning rate: {}, avg test score: {}.\n\n".format(self.config['lr'], test_results['metric']))

        print('Test results:', test_results)
        return test_results, os.path.join(self.save_path, 'parameter.pkl'), epoch

    def test(self, dataloader):
        loss_fn = nn.BCEWithLogitsLoss()
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        avg_loss = Averager()

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'], data_type=self.config['data_type'], model_name=self.config['model_name'])
                labels = batch_data['label'].float().unsqueeze(1)

                logits = self.model(
                    batch_data['content'],
                    batch_data['content_masks'],
                    batch_data.get('td_rationale'),
                    batch_data.get('td_rationale_masks'),
                    batch_data.get('cs_rationale'),
                    batch_data.get('cs_rationale_masks')
                )

                loss = loss_fn(logits, labels)
                avg_loss.add(loss.item())

                probs = torch.sigmoid(logits)
                pred.extend(probs.cpu().numpy().tolist())
                label.extend(labels.cpu().numpy().tolist())

        metrics_result = metrics(label, pred)
        aux_info = {
            'val_avg_loss': avg_loss.item()
        }

        return metrics_result, aux_info

    def predict(self, dataloader):
        if self.config.get('eval_mode', False):
            self.model = self.initialize_model()
            if self.config['use_cuda']:
                self.model = self.model.cuda()
            print('========== In test process ==========')
            print('Now loading the test model...')
            self.model.load_state_dict(torch.load(self.config['eval_model_path']))

        pred = []
        label = []
        id_list = []
        ae = []
        accuracy = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'], data_type=self.config['data_type'], model_name=self.config['model_name'])
                labels = batch_data['label'].float().unsqueeze(1)
                ids = batch_data.get('id', [])

                logits = self.model(
                    batch_data['content'],
                    batch_data['content_masks'],
                    batch_data.get('td_rationale'),
                    batch_data.get('td_rationale_masks'),
                    batch_data.get('cs_rationale'),
                    batch_data.get('cs_rationale_masks')
                )

                probs = torch.sigmoid(logits)
                batch_pred = probs.cpu().numpy().tolist()
                batch_label = labels.cpu().numpy().tolist()
                batch_id = ids

                label.extend(batch_label)
                pred.extend(batch_pred)
                id_list.extend(batch_id)

                # Calculation of absolute errors and accuracy
                ae_list = [abs(p[0] - l[0]) for p, l in zip(batch_pred, batch_label)]
                ae.extend(ae_list)
                accuracy_list = [1 if ae < 0.5 else 0 for ae in ae_list]
                accuracy.extend(accuracy_list)

        metrics_result = metrics(label, pred)
        return metrics_result, label, pred, id_list, ae, accuracy
