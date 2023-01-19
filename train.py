import torch
import argparse
from datasets import QQRDataset,QQR_data
from tqdm import tqdm
from gensim.models import KeyedVectors
import time
from torch.utils.data import DataLoader
from models import SemNN,SemLSTM,SemAttention
import os
import torch.nn as nn
import torch.optim as optim

def train(args):
    batch_size = args.batch_size
    lr = args.lr
    save_path = args.savepath
    data_dir = args.datadir
    w2v_path = args.w2v_path
    max_length = args.max_length
    epochs = args.epochs
    model_name = args.model_name
    dropout_prob = args.dropout_prob
    in_feat = args.in_feat
    
    begin_time = time.perf_counter()
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path,binary=False)
    end_time = time.perf_counter()
    print("Load {} cost {:.2f}s".format(w2v_path,end_time-begin_time))
    w2v_map = w2v_model.key_to_index
    
    
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    save_path = os.path.join(save_path,model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data = QQR_data(data_dir)
    train_dataset = QQRDataset(data.get_train_data(),data.get_labels(),w2v_map=w2v_map,max_length=max_length)
    val_dataset = QQRDataset(data.get_dev_data(),data.get_labels(),w2v_map=w2v_map,max_length=max_length)
    test_dataset = QQRDataset(data.get_test_data,data.get_labels(),w2v_map=w2v_map,max_length=max_length)
    
    train_examples_num = len(train_dataset)
    val_examples_num = len(val_dataset)
    
    dataset = {'train':train_dataset,'val':val_dataset,'test':test_dataset}
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
    print(model)
    
    model_paramters = model.parameters()
    print('Model Name: '+model_name)
    print('Total params: %.2fM' % (sum(p.numel() for p in model_paramters) / 1000000.0))
    
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    model.to(device)
    
    
    optimizer = optim.SGD(model_paramters, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)
    best_val_acc = 0.0
    for epoch in range(epochs):
        for phase in ['train','val']:
            runing_loss = 0.0
            running_corrects = 0.0
            
            #Train or eval
            if(phase=='train'):
                optimizer.step()
                model.train()
            else:
                model.eval()
            
            dataloader = DataLoader(dataset[phase],batch_size=batch_size,shuffle=True,num_workers=4)
            for text_example in tqdm(dataloader):
                text_a_inputs_id = text_example["text_a_inputs_id"].to(device)
                text_b_inputs_id = text_example["text_b_inputs_id"].to(device)
                text_a_attention_mask = text_example["text_a_attention_mask"].to(device)
                text_b_attention_mask = text_example["text_b_attention_mask"].to(device)
                labels = text_example['labels'].to(device)
                
                optimizer.zero_grad()
                
                if(phase=='train'):
                    outputs = model(text_a_inputs_id,text_b_inputs_id,text_a_attention_mask,text_b_attention_mask)
                else:
                    with torch.no_grad():
                        outputs = model(text_a_inputs_id,text_b_inputs_id,text_a_attention_mask,text_b_attention_mask)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs,1)[1]
                # print(preds.sum())
                
                loss = criterion(outputs,labels)
                
                if(phase=='train'):
                    loss.backward()
                    optimizer.step()
                
                runing_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds==labels.data)
            
            epoch_loss = runing_loss/len_dataset.get(phase)
            epoch_acc = running_corrects.double()/len_dataset.get(phase)
            if(phase=='val'):
                if(best_val_acc<epoch_acc):
                    best_val_acc = epoch_acc
                    torch.save({
                        'epoch':epoch+1,
                        'state_dict':model.state_dict(),
                        'opt_dict':optimizer.state_dict()
                    },os.path.join(save_path,'best_model.pth.tar'))
            
            if(epoch==epochs-1):
                torch.save({
                        'epoch':epoch+1,
                        'state_dict':model.state_dict(),
                        'opt_dict':optimizer.state_dict()
                    },os.path.join(save_path,'lastest_model.pth.tar'))
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, epochs, epoch_loss, epoch_acc))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    
    parse.add_argument('--model_name',type=str,default="SemAttention",help="Model name for train [SemNN,SemLSTM,SemAttention]")
    
    parse.add_argument('--batch_size',type=int,default=8,help="Batch-size for train")
    
    parse.add_argument('--in_feat',type=int,default=100,help="Length of features for embbeding word")
    
    parse.add_argument('--max_length',type=int,default=32,help="Max length for setence")
    
    parse.add_argument('--epochs',type=int,default=50,help="Set epochs for train")
    
    parse.add_argument('--lr',type=float,default=1e-3,help="Learning Rate for train")
    
    parse.add_argument('--dropout_prob',type=float,default=0.1,help="Dropout ratio for dropout layers")
    
    parse.add_argument('--savepath',type=str,default="./results",help="Save dir for trained model")
    
    parse.add_argument('--datadir',type=str,default='./data',help="Data path for train")
    
    parse.add_argument('--gpu',type=str,default='1',help="Gpu id for train")
    
    parse.add_argument('--w2v_path',type=str,default='./tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt',help="Path for w2v_model file")
    
    args = parse.parse_args()
    
    train(args)