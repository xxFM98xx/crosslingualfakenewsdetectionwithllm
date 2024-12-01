import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import random
import pandas as pd
import json
import numpy as np
import nltk
import jieba
from transformers import BertTokenizer,XLMRobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

label_dict = {
    "real": 0,
    "fake": 1,
    0: 0,
    1: 1
}

label_dict_ftr_pred = {
    "real": 0,
    "fake": 1,
    "other": 2,
    0: 0,
    1: 1,
    2: 2
}

def word2input(texts, max_len, tokenizer):
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def get_dataloader(path, max_len, batch_size, shuffle, bert_path, data_type, language):

    tokenizer = XLMRobertaTokenizer.from_pretrained(bert_path)

    if data_type == 'rationale':
        data_list = json.load(open(path, 'r',encoding='utf-8'))
        df_data = pd.DataFrame(columns=('content','label'))
        for item in data_list:
            tmp_data = {}

            # content info
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item['source_id']

            tmp_data['FTR_2'] = item['td_rationale']
            tmp_data['FTR_3'] = item['cs_rationale']

            tmp_data['FTR_2_pred'] = item['td_pred']
            tmp_data['FTR_3_pred'] = item['cs_pred']

            tmp_data['FTR_2_acc'] = item['td_acc']
            tmp_data['FTR_3_acc'] = item['cs_acc']

            df_data = df_data.append(tmp_data, ignore_index=True)

        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        id = torch.tensor(df_data['id'].to_numpy())

        FTR_2_pred = torch.tensor(df_data['FTR_2_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())
        FTR_3_pred = torch.tensor(df_data['FTR_3_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())

        FTR_2_acc = torch.tensor(df_data['FTR_2_acc'].astype(int).to_numpy())
        FTR_3_acc = torch.tensor(df_data['FTR_3_acc'].astype(int).to_numpy())

        FTR_2 = df_data['FTR_2'].to_numpy()
        FTR_3 = df_data['FTR_3'].to_numpy()

        content_token_ids, content_masks = word2input(content, max_len, tokenizer)
        
        FTR_2_token_ids, FTR_2_masks = word2input(FTR_2, max_len, tokenizer)
        FTR_3_token_ids, FTR_3_masks = word2input(FTR_3, max_len, tokenizer)

        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                FTR_2_pred,
                                FTR_2_acc,
                                FTR_3_pred,
                                FTR_3_acc,
                                FTR_2_token_ids,
                                FTR_2_masks,
                                FTR_3_token_ids,
                                FTR_3_masks,
                                label,
                                id,
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=shuffle
        )
        return dataloader
    else:
        print('No match data type!')
        exit()


def get_dataloader_baseline(data_path, max_len, batch_size, shuffle, bert_path, data_type, language, use_rationales=True):
    tokenizer = XLMRobertaTokenizer.from_pretrained(bert_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    contents = []
    td_rationales = []
    cs_rationales = []
    labels = []
    ids = []

    for item in data:
        contents.append(item['content'])
        if use_rationales:
            td_rationales.append(item['td_rationale'])
            cs_rationales.append(item['cs_rationale'])
        labels.append(item['label'])
        ids.append(item['source_id'])  # Use 'source_id' as ID

    # Tokenisation of content
    content_ids, content_masks = word2input(contents, max_len, tokenizer)

    if use_rationales:
        # Tokenisation of rationales
        td_rationale_ids, td_rationale_masks = word2input(td_rationales, max_len, tokenizer)
        cs_rationale_ids, cs_rationale_masks = word2input(cs_rationales, max_len, tokenizer)

    # Labels
    label_dict = {
        "real": 0,
        "fake": 1,
        0: 0,
        1: 1
    }
    labels = torch.tensor([label_dict[label] for label in labels])
    ids = torch.tensor(ids)

    if use_rationales:
        dataset = TensorDataset(
            content_ids,
            content_masks,
            td_rationale_ids,
            td_rationale_masks,
            cs_rationale_ids,
            cs_rationale_masks,
            labels,
            ids
        )
    else:
        dataset = TensorDataset(
            content_ids,
            content_masks,
            labels,
            ids
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=False,
        num_workers=1
    )

    return dataloader

