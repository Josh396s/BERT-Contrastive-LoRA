import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

class ScenarioModel(nn.Module):
    def __init__(self, args, tokenizer, target_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_setup(args)
        self.target_size = target_size

        # task1: add necessary class variables as you wish.
        
        # task2: initilize the dropout and classify layers
        self.dropout = nn.Dropout(args.drop_rate)
        self.classify = Classifier(args, target_size)
    
    def model_setup(self, args):
        print(f"Setting up {args.model} model")

        # task1: get a pretrained model of 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
    
        self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

    def forward(self, inputs, targets=None):
        """
        task1: 
            feeding the input to the encoder, 
        task2: 
            take the last_hidden_state's <CLS> token as output of the
            encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
        task3:
            feed the output of the dropout layer to the Classifier which is provided for you.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
    
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask) 

        cls_token = outputs.last_hidden_state[:, 0, :]  
        
        cls_token = self.dropout(cls_token)
        
        logits = self.classify(cls_token)

        return logits


class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit

class ClassifierModel(nn.Module):
    def __init__(self, encoder, target_dim=10):
        super(ClassifierModel, self).__init__()
        self.encoder = encoder  # Reuse the frozen encoder
        self.classifier = nn.Linear(encoder.output_dim, target_dim)  # New classifier head

    def forward(self, x):
        features = self.encoder(x)  # Get features from the frozen encoder
        logits = self.classifier(features)  # Classify
        return logits


class CustomModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size):
    super().__init__(args, tokenizer, target_size)
    
    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model
    self.pooler = nn.Linear(args.embed_dim, args.embed_dim)
    self.pooler_activation = nn.Tanh()
    self.regressor = nn.Linear(args.embed_dim, target_size)

    self.optimizer = self.initialize_llrd_optimizer(args)

  # Technique 1: Initialize optimizer for Layer-wise Learning Rate Decay (LLRD)
  def initialize_llrd_optimizer(self, args):
    # Learning rates for each layer
    head_lr = 3.6e-6
    top_lr = 3.5e-6
    # decay_rate = 0.9

    # head params
    head_params = {
        "params": list(self.pooler.parameters()) + list(self.regressor.parameters()),
        "lr": head_lr,
        "weight_decay": args.weight_decay
    }

    # layer params
    layer_params = []
    layers = list(self.encoder.encoder.layer)
    num_layers = len(layers)

    for idx, layer in enumerate(reversed(layers)): # i=0 corresponds to top(last)
        # current_lr = top_lr * (decay_rate ** idx)
        layer_params.append({
            "params": layer.parameters(),
            "lr": top_lr, # current_lr,
            "weight_decay": args.weight_decay
        })
    
    # embed params
    embed_params = {
        "params": self.encoder.embeddings.parameters(),
        "lr": top_lr, # * (decay_rate ** num_layers),
        "weight_decay": args.weight_decay
    }
    # combined optimizer
    combined_params = [head_params] + layer_params + [embed_params]
    optimizer = torch.optim.AdamW(combined_params)
    return optimizer
  
  # Technique 2: Warm up scheduler
  def initialize_scheduler (self, train_dataset, batch_size, epochs):
    train_steps = len(train_dataset) // batch_size * epochs
    num_warmup_steps = int(train_steps * 0.1)
    
    def lr_lambda(curr_step):
      return min((curr_step + 1) / num_warmup_steps, 1.0) if curr_step < num_warmup_steps else 1.0

    scheduler = LambdaLR(self.optimizer, lr_lambda)

    min_lr = 1e-6
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = max(param_group['lr'], min_lr)
    return scheduler
  
  # Forward for llrd
  def forward(self, inputs, targets=None):
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    input_ids = input_ids.squeeze(1)
    attention_mask = attention_mask.squeeze(1)
    
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask) 

    cls_token = outputs.last_hidden_state[:, 0, :]  
    
    pooled_output = self.pooler(cls_token)
    pooled_output = self.pooler_activation(pooled_output)

    pooled_output = self.dropout(pooled_output)
    
    logits = self.regressor(pooled_output)

    return logits
  
  

class SupConModel(ScenarioModel):
    def __init__(self, args, tokenizer, target_size, feat_dim=768):
        super().__init__(args, tokenizer, target_size)

        # Projection Head (Linear Layer)
        self.head = nn.Linear(feat_dim, feat_dim)

        # Dropout layer with argparse-defined dropout rate
        self.dropout = nn.Dropout(args.drop_rate)

    def forward(self, inputs):
        """
        Args:
            inputs: Dictionary containing input tensors (input_ids, attention_mask)

        Returns:
            embeddings: Normalized embeddings after projection head
        """
        inputs = {key: val.squeeze(1) if val.dim() == 3 else val for key, val in inputs.items()}

        outputs = self.encoder(**inputs)

        # Extract <CLS> token from last_hidden_state
        cls_embedding = outputs.last_hidden_state[:, 0]  # [batch_size, feat_dim]

        cls_embedding = self.dropout(cls_embedding)

       
        embeddings = F.normalize(self.head(cls_embedding), dim=-1)

        return embeddings 
