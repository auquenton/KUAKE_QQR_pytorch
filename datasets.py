#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :datasets.py
@Description  :DataSets  for NLP_query
@Time         :2023/01/17 15:21:45
@Author       :KangQing
@Version      :1.0
'''


import sys
sys.path.append('./')
import json
import os
import jieba
from torch.utils.data import Dataset
import torch
import numpy as np
import logging

jieba.setLogLevel(logging.INFO)


class InputExample():
    def __init__(self,id:str,text_a:str,text_b:str=None,label:str=None) -> None:  
        self.id = id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
    def __str__(self) -> str:
        return json.dumps({'id':self.id,'text_a':self.text_a,'text_b':self.text_b,'label':self.label},indent=2,ensure_ascii=False)+'\n'


class QQR_data():
    def __init__(self,data_path='data') -> None:
        self.data_path = data_path
    
    def get_data(self,json_data_path):
        with open(json_data_path,'r',encoding='utf-8') as f:
            data = json.load(f,encoding='utf-8')
        
        examples = []
        for example in data:
            examples.append(InputExample(example['id'],example['query1'],example['query2'],example['label'] if example['label']!="" else None))
        return examples

    def get_labels(self):
        return ['0','1','2']
    
    def get_train_data(self):
        path = os.path.join(self.data_path,'KUAKE-QQR_{}.json'.format('train'))
        return self.get_data(path)
    
    def get_dev_data(self):
        path = os.path.join(self.data_path,'KUAKE-QQR_{}.json'.format('dev'))
        return self.get_data(path)
    
    def get_test_data(self):
        path = os.path.join(self.data_path,'KUAKE-QQR_{}.json'.format('test'))
        return self.get_data(path)
    

    
class QQRDataset(Dataset):
    def __init__(self,examples_list,labels_list,w2v_map,max_length):
        self.examples_list = examples_list
        self.label2id = {label:idx for idx,label in enumerate(labels_list)}
        self.id2label = {idx:label for idx,label in enumerate(labels_list)}
        self.w2v_map = w2v_map
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples_list)
    
    def _tokenize(self,text):
        token_list = list(jieba.cut(text))
        token_ids = []
        for token in token_list:
            if(token in self.w2v_map):
                token_ids.append(self.w2v_map.get(token))
            else:
                if(len(token)>1):
                    for character in token:
                        token_ids.append(self.w2v_map.get(token) if self.w2v_map.get(token)!=None else np.random.choice(len(self.w2v_map),1).item())
                else:
                    token_ids.append(np.random.choice(len(self.w2v_map),1).item())
        
        token_ids,attention_mask = self._pad_and_cut(token_ids)
        return token_ids,attention_mask
    
    def _pad_and_cut(self,token_ids):
        
        #Generate attention mask
        attention_mask = None
        
        if(len(token_ids)>self.max_length):
            token_ids = token_ids[:self.max_length]
            attention_mask = [1]*self.max_length
        else:
            attention_mask = [1]*len(token_ids)
            diff = self.max_length - len(token_ids)
            token_ids.extend([0]*diff)
            attention_mask.extend([0]*diff)
        
        return torch.tensor(token_ids,dtype=torch.long),torch.tensor(attention_mask,dtype=torch.long)
    
    
    def __getitem__(self,index):
        example = self.examples_list[index]
        idx = example.id
        text_a = example.text_a
        text_b = example.text_b
        if(example.label in self.label2id):
            label = self.label2id[example.label]
        else:
            label = 3
        
        text_a_inputs_id,text_a_attention_mask = self._tokenize(text_a)
        text_b_inputs_id,text_b_attention_mask = self._tokenize(text_b)
        
        label = torch.tensor(label,dtype=torch.long)

        
        return {
            'text_a_inputs_id':text_a_inputs_id,
            'text_b_inputs_id':text_b_inputs_id,
            'text_a_attention_mask':text_a_attention_mask,
            'text_b_attention_mask':text_b_attention_mask,
            'labels':label,
            'text_a':text_a,
            'text_b':text_b,
            'idx':idx
        }
        
    

