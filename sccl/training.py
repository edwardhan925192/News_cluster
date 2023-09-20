import os
import time
import numpy as np
from sklearn import cluster

from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader

import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss

class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)
        
        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")
        
    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)

            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
            
        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)
            
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.cuda(), attention_mask.cuda()
        
        
    def train_step_virtual(self, input_ids, attention_mask):
        embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]
        
        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += 0.5*cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    
    def train_step_explicit(self, input_ids, attention_mask):
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()
            
            cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0]
            loss += cluster_loss
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    
    
    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()
        for i in np.arange(self.args.max_iter+1):
            try:
                batch = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)

            losses = self.train_step_virtual(input_ids, attention_mask) if self.args.augtype == "virtual" else self.train_step_explicit(input_ids, attention_mask)

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                # Removed the evaluation embedding

        return None
