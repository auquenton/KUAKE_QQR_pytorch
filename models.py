import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class LstmAttEncoder(nn.Module):
    def __init__(self, in_feat: int = 100):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=in_feat, bidirectional=True, batch_first=True)
    def forward(self, token_embeds, attention_mask):
        batch_size = attention_mask.size(0)
        output, (h, c) = self.lstm(token_embeds)
        output, lens_output = pad_packed_sequence(output, batch_first=True)
        
        return output,lens_output


class LstmDecoder(nn.Module):

    def __init__(self, in_feat: int = 100, dropout_prob: float = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=in_feat, bidirectional=True, batch_first=True)

    def forward(self, token_embeds, attention_mask):
        batch_size = attention_mask.size(0)
        output, (h, c) = self.lstm(token_embeds)
        output, lens_output = pad_packed_sequence(output, batch_first=True)   # [B, L, H]
        output = torch.stack([torch.mean(output[i][:lens_output[i]], dim=0) for i in range(batch_size)], dim=0)
        
        return output   
    
class Encoder(nn.Module):
    def __init__(self,in_feat=100,dropout_prob=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_feat,in_feat)
        self.linear2 = nn.Linear(in_feat,in_feat)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self,token_embeds,attention_mask):
        batch_size = token_embeds.size(0)
        
        x = torch.stack([token_embeds[i,attention_mask[i,:],:].sum(dim=0) for i in range(batch_size)],dim=0)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(self.dropout(x)))
        
        return x

class LstmEncoder(nn.Module):

    def __init__(self, in_feat: int = 100, dropout_prob: float = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=in_feat, bidirectional=True, batch_first=True)

    def forward(self, token_embeds, attention_mask):
        batch_size = attention_mask.size(0)
        output, (h, c) = self.lstm(token_embeds)
        output, lens_output = pad_packed_sequence(output, batch_first=True)
        # 双向LSTM出来的hidden states做平均
        output = torch.stack([output[i][:lens_output[i]].mean(dim=0) for i in range(batch_size)], dim=0)
        return output


class Classifier(nn.Module):
    def __init__(self, in_feat, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(in_feat, in_feat // 2)
        self.dense2 = nn.Linear(in_feat // 2, num_labels)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.act(self.dense1(self.dropout(x)))
        x = self.dense2(self.dropout(x))
        return x     
    
class SemNN(nn.Module):
    def __init__(self,
        in_feat = 100,
        num_labels = 3,
        dropout_prob = 0.1,
        w2v_mapping = None,
        vocab_size = None,
        word_embedding_dim = None
    ):
        super().__init__()
        self.num_labels = num_labels
        self._init_word_embedding(w2v_mapping,vocab_size,word_embedding_dim)
        self.encoder = Encoder(in_feat=in_feat)
        self.classifier = Classifier(in_feat=2*in_feat,num_labels=num_labels,dropout_prob=dropout_prob)
        
    def _init_word_embedding(self,state_dict=None,vocab_size=None,word_embedding_dim=None):
        if state_dict is None:
            self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        else:
            state_dict = torch.tensor(state_dict.vectors, dtype=torch.float32)
            state_dict[0] = torch.zeros(state_dict.size(-1))
            self.word_embedding = nn.Embedding.from_pretrained(state_dict, freeze=True, padding_idx=0)
    
    def forward(self,
                text_a_inputs_id,
                text_b_inputs_id,
                text_a_attention_mask,
                text_b_attention_mask):
        
        text_a_vec = self.word_embedding(text_a_inputs_id)
        text_b_vec = self.word_embedding(text_b_inputs_id)
        
        text_a_vec = self.encoder(text_a_vec,text_a_attention_mask)
        text_b_vec = self.encoder(text_b_vec,text_b_attention_mask)
        
        pooler_output = torch.cat([text_a_vec,text_b_vec],dim=-1)
        logits = self.classifier(pooler_output)
        
        return logits


class CrossAttention(nn.Module):
    def __init__(self,in_feat,dropout_prob):
        super().__init__()
        self.dense = nn.Linear(4*in_feat,in_feat//2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        
    
    def forward(self,a,b,mask_a,mask_b):
        in_feat = a.size(-1)
        
        # a:[B,L1,H] b:[B,L2,H]
        
        # attention score [B,L1,L2]
        cross_attn = torch.matmul(a,b.transpose(1,2))
        
        # ignore b(L2) padding information [B,L1,L2]
        row_attn = cross_attn.masked_fill((mask_b==False).unsqueeze(1),-1e9)
        row_attn = row_attn.softmax(dim=2) #[B,L1,L2]
        
        # ignore a(L1) padding information
        col_attn = cross_attn.permute(0,2,1).contiguous() #[B,L2,L1]
        col_attn = col_attn.masked_fill((mask_a==False).unsqueeze(1),-1e9)
        col_attn = col_attn.softmax(dim=2) #[B,L2,L1]
        
        #attention score * value
        att_a = torch.matmul(row_attn,b) #[B, L1, H]
        att_b = torch.matmul(col_attn,a) #[B, L2, H]
        
        diff_a = a - att_a
        diff_b = b - att_b
        prod_a = a * att_a
        prod_b = b * att_b
        
        #Cat
        a = torch.cat([a,att_a,diff_a,prod_a],dim=-1)   #[B,L1,4H]
        b = torch.cat([b,att_b,diff_b,prod_b],dim=-1)   #[B,L2,4H]
        
        a = self.act(self.dense(self.dropout(a))) #[B,L1,H/2]
        b = self.act(self.dense(self.dropout(b))) #[B,L2,H/2]
        
        return a,b
        


class SemLSTM(nn.Module):
    def __init__(self,
        in_feat = 100,
        num_labels = 3,
        dropout_prob = 0.1,
        w2v_mapping = None,
        vocab_size = None,
        word_embedding_dim = None
    ):
        super().__init__()
        self.num_labels = num_labels
        self._init_word_embedding(w2v_mapping,vocab_size,word_embedding_dim)
        self.encoder = LstmEncoder(in_feat=in_feat)
        self.classifier = Classifier(in_feat=4*in_feat,num_labels=num_labels,dropout_prob=dropout_prob)
        
    def _init_word_embedding(self,state_dict=None,vocab_size=None,word_embedding_dim=None):
        if state_dict is None:
            self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        else:
            state_dict = torch.tensor(state_dict.vectors, dtype=torch.float32)
            state_dict[0] = torch.zeros(state_dict.size(-1))
            self.word_embedding = nn.Embedding.from_pretrained(state_dict, freeze=True, padding_idx=0)
    
    def forward(self,
                text_a_inputs_id,
                text_b_inputs_id,
                text_a_attention_mask,
                text_b_attention_mask):
        
        #Embedding
        text_a_vec = self.word_embedding(text_a_inputs_id)
        text_b_vec = self.word_embedding(text_b_inputs_id)
        
        #Pack
        text_a_vec = pack_padded_sequence(text_a_vec,text_a_attention_mask.cpu().long().sum(dim=-1),batch_first=True,enforce_sorted=False)
        text_b_vec = pack_padded_sequence(text_b_vec,text_b_attention_mask.cpu().long().sum(dim=-1),batch_first=True,enforce_sorted=False)
        
        #LSTM
        text_a_vec = self.encoder(text_a_vec,text_a_attention_mask)
        text_b_vec = self.encoder(text_b_vec,text_b_attention_mask)
        
        #Cat
        pooler_output = torch.cat([text_a_vec,text_b_vec],dim=-1)
        logits = self.classifier(pooler_output)
        
        return logits
      
class SemAttention(nn.Module):
    def __init__(self,
        in_feat = 100,
        num_labels = 3,
        dropout_prob = 0.1,
        w2v_mapping = None,
        vocab_size = None,
        word_embedding_dim = None
    ):
        super().__init__()
        self.num_labels = num_labels
        self._init_word_embedding(w2v_mapping,vocab_size,word_embedding_dim)
        self.encoder = LstmAttEncoder(in_feat=in_feat)
        self.classifier = Classifier(in_feat=4*in_feat,num_labels=num_labels,dropout_prob=dropout_prob)
        self.crossattention = CrossAttention(in_feat=2*in_feat,dropout_prob=dropout_prob)
        self.decoder = LstmDecoder(in_feat=in_feat,dropout_prob=dropout_prob)
        
    def _init_word_embedding(self,state_dict=None,vocab_size=None,word_embedding_dim=None):
        if state_dict is None:
            self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        else:
            state_dict = torch.tensor(state_dict.vectors, dtype=torch.float32)
            state_dict[0] = torch.zeros(state_dict.size(-1))
            self.word_embedding = nn.Embedding.from_pretrained(state_dict, freeze=True, padding_idx=0)
    
    def forward(self,
                text_a_inputs_id,
                text_b_inputs_id,
                text_a_attention_mask,
                text_b_attention_mask):
        
        #Embedding
        text_a_vec = self.word_embedding(text_a_inputs_id) #[B,L1,H]
        text_b_vec = self.word_embedding(text_b_inputs_id) #[B,L2,H]
        
        #Pack
        text_a_vec = pack_padded_sequence(text_a_vec,text_a_attention_mask.cpu().long().sum(dim=-1),batch_first=True,enforce_sorted=False)
        text_b_vec = pack_padded_sequence(text_b_vec,text_b_attention_mask.cpu().long().sum(dim=-1),batch_first=True,enforce_sorted=False)
        text_a_attention_mask = pack_padded_sequence(text_a_attention_mask,text_a_attention_mask.cpu().long().sum(dim=-1),batch_first=True,enforce_sorted=False)
        text_b_attention_mask = pack_padded_sequence(text_b_attention_mask,text_b_attention_mask.cpu().long().sum(dim=-1),batch_first=True,enforce_sorted=False)
        text_a_attention_mask,_ = pad_packed_sequence(text_a_attention_mask,batch_first=True)
        text_b_attention_mask,_ = pad_packed_sequence(text_b_attention_mask,batch_first=True)
        
        #LSTM_Encoder
        text_a_vec,text_a_len = self.encoder(text_a_vec,text_a_attention_mask) #[B,L1,2H]
        text_b_vec,text_b_len = self.encoder(text_b_vec,text_b_attention_mask) #[B,L2,2H]
        
        #cross attention
        text_a_vec,text_b_vec = self.crossattention(text_a_vec,text_b_vec,text_a_attention_mask,text_b_attention_mask) #[B,L1,H]
        text_a_vec = pack_padded_sequence(text_a_vec,text_a_len,batch_first=True,enforce_sorted=False)
        text_b_vec = pack_padded_sequence(text_b_vec,text_b_len,batch_first=True,enforce_sorted=False)
        
        #Decoder
        text_a_vec = self.decoder(text_a_vec,text_a_attention_mask)
        text_b_vec = self.decoder(text_b_vec,text_b_attention_mask)
        
        #Cat
        pooler_output = torch.cat([text_a_vec,text_b_vec],dim=-1)
        logits = self.classifier(pooler_output)
        
        return logits