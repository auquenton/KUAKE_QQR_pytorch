import torch
import argparse
from datasets import QQRDataset,QQR_data,BertClassificationDataset
from tqdm import tqdm
from gensim.models import KeyedVectors
import time
from torch.utils.data import DataLoader
from models import SemNN,SemLSTM,SemAttention
import os
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

model_type1_list = ['SemNN','SemAttention','SemLSTM']
model_type2_list = ['Bert']



def train(args):
    batch_size = args.batch_size
    data_dir = args.datadir
    w2v_path = args.w2v_path
    max_length = args.max_length
    model_name = args.model_name
    in_feat = args.in_feat
    dropout_prob = args.dropout_prob
    model_path = args.model_path
    
    if model_name in model_type1_list:
        begin_time = time.perf_counter()
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path,binary=False)
        end_time = time.perf_counter()
        print("Load {} cost {:.2f}s".format(w2v_path,end_time-begin_time))
        w2v_map = w2v_model.key_to_index
        
    elif model_name in model_type2_list:
        tokenizer = AutoTokenizer.from_pretrained(w2v_path)
    
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    data = QQR_data(data_dir)
    
    if model_name in model_type1_list:
        train_dataset = QQRDataset(data.get_train_data(),data.get_labels(),w2v_map=w2v_map,max_length=max_length)
        val_dataset = QQRDataset(data.get_dev_data(),data.get_labels(),w2v_map=w2v_map,max_length=max_length)
       
    elif model_name in model_type2_list:
        train_dataset = BertClassificationDataset(data.get_train_data(),tokenizer=tokenizer,label_list=data.get_labels(),max_length=max_length)
        val_dataset = BertClassificationDataset(data.get_dev_data(),tokenizer=tokenizer,label_list=data.get_labels(),max_length=max_length)
    
    train_examples_num = len(train_dataset)
    val_examples_num = len(val_dataset)
    
    dataset = {'train':train_dataset,'val':val_dataset}
    len_dataset = {'train':train_examples_num,'val':val_examples_num}
    
    if model_name == "SemNN":
        model = SemNN(
            in_feat=in_feat,
            num_labels=len(data.get_labels()),
            dropout_prob=dropout_prob,
            w2v_mapping=w2v_model
        )
    elif model_name == "SemLSTM":
        model = SemLSTM(in_feat=in_feat,
                        num_labels=len(data.get_labels()),
                        dropout_prob=dropout_prob,
                        w2v_mapping=w2v_model)
    elif model_name == "SemAttention":
        model = SemAttention(
            in_feat=in_feat,
            num_labels = len(data.get_labels()),
            dropout_prob=dropout_prob,
            w2v_mapping=w2v_model
        )
    elif model_name == "Bert":
        model = BertForSequenceClassification.from_pretrained(w2v_path,num_labels=len(data.get_labels()))
     
    print(model)
    
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    print('Model Name: '+model_name)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    best_val_acc = 0.0
    for phase in ['train','val']:
        runing_loss = 0.0
        running_corrects = 0.0
        
        model.eval()
        
        dataloader = DataLoader(dataset[phase],batch_size=batch_size,shuffle=True,num_workers=4)
        for text_example in tqdm(dataloader):
            if model_name in model_type1_list:
                text_a_inputs_id = text_example["text_a_inputs_id"].to(device)
                text_b_inputs_id = text_example["text_b_inputs_id"].to(device)
                text_a_attention_mask = text_example["text_a_attention_mask"].to(device)
                text_b_attention_mask = text_example["text_b_attention_mask"].to(device)
            elif model_name in model_type2_list:
                input_ids = text_example.get('input_ids').to(device)
                token_type_ids = text_example.get('token_type_ids').to(device)
                attention_mask = text_example.get('attention_mask').to(device)
                
            labels = text_example['labels'].to(device)
            

            with torch.no_grad():
                if model_name in model_type1_list:
                    outputs = model(text_a_inputs_id,text_b_inputs_id,text_a_attention_mask,text_b_attention_mask)
                elif model_name in model_type2_list:
                    outputs = model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,return_dict=True).get('logits')
                        
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs,1)[1]

            running_corrects += torch.sum(preds==labels.data)
        
        epoch_acc = running_corrects.double()/len_dataset.get(phase)
        print("[{}]  Acc: {}".format(phase, epoch_acc))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    
    parse.add_argument('--model_name',type=str,default="SemAttention",help="Model name for train [SemNN,SemLSTM,SemAttention,Bert]")
    
    parse.add_argument('--batch_size',type=int,default=8,help="Batch-size for train")
    
    parse.add_argument('--in_feat',type=int,default=100,help="Length of features for embbeding word")
    
    parse.add_argument('--model_path',type=str,default='./results/SemAttention/best_model.pth.tar',help="Saved model path")
    
    parse.add_argument('--max_length',type=int,default=32,help="Max length for setence")
    
    parse.add_argument('--dropout_prob',type=float,default=0.1,help="Dropout ratio for dropout layers")
    
    parse.add_argument('--datadir',type=str,default='./data',help="Data path for train")
    
    parse.add_argument('--gpu',type=str,default='1',help="Gpu id for train")
    
    parse.add_argument('--w2v_path',type=str,default='./tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt',help="Path for w2v_model file")
    
    args = parse.parse_args()
    
    train(args)