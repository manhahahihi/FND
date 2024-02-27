import torch.nn as nn
from config import *
from transformers import AutoModel
import sys





# Define your model (simple binary classification model)
class FakeNewsClassifier(torch.nn.Module):
    def __init__(self, hidden_size, dense_size, num_labels):
        super().__init__()

        # Load PhoBERT 
        self.phobert = AutoModel.from_pretrained(name_model)

        # Freezing the parameters and defining trainable BERT structure
        for name, param in self.phobert.named_parameters():
            if 'pooler' not in name:
                param.requires_grad = False
            # param.requires_grad = False

        # Define finetune layers
        self.dense = nn.Linear(hidden_size, dense_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(dense_size, num_labels)

    def forward(self, input_ids, attention_mask):
        
        # take features from pretrained PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        # outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:, 0, :]
        
        # put to fine tunning layers
        x = self.dropout(outputs)
        x = self.dense(x) # fc1
        x = nn.ReLU()(x) # activation func
        x = self.dropout(x)
        x = self.out_proj(x) # fc2 

        return x