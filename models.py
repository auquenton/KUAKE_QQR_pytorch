import torch
import torch.nn as nn
import numpy as np

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
        
