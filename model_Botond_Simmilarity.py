import copy
from functools import reduce

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report

from collections import defaultdict
from torch.nn.functional import binary_cross_entropy, cross_entropy
import math
import random
from utils.logic import ConceptEmbedding

## VRAGEN
# waarom projecten we rule embedding nog eens met NN?, waarom niet gewoon enkel input embedding?


class MNISTModel(pl.LightningModule):

    def initialize_rule_selector(self):
        if self.initialized:
            def initialize_weights(module):
                if isinstance(module, torch.nn.Linear):
                    module.reset_parameters()
            self.neural_rule_selector.apply(initialize_weights)
            return
        self.initialized = True

        if self.effective_n_rules > self.rule_mask.shape[1]:
            self.rule_mask = torch.cat([self.rule_mask, torch.ones(self.n_tasks, self.effective_n_rules - self.rule_mask.shape[1])], dim=-1)

        # === Similarity Selector ===

        # Waarom doen we NN op embeddings van rules?
        self.rule_emb_proj = torch.nn.Sequential(
            torch.nn.Linear(self.rule_emb_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        ).to(self.device)

        self.input_emb_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        ).to(self.device)

    def compute_rule_logits(self, input_emb):
        """
        input_emb:    (batch, embedding_size)
        returns logits: (batch, n_tasks, effective_n_rules)
        """
        # 2. Get rule embeddings and project to keys
        r = self.rule_module.rules.weight                       # (n_tasks * n_rules, rule_emb_size)
        r = r.view(self.n_tasks, self.effective_n_rules, self.rule_emb_size)
        r = self.rule_emb_proj(r)                               # (n_tasks, n_rules, emb)

        # 3. Cosine similarity between query and each rule embedding
        x = self.input_emb_proj(input_emb)
        x = x.unsqueeze(1).unsqueeze(1)                 # (batch, 1, 1, emb)
        r = r.unsqueeze(0)                                      # (1, n_tasks, n_rules, emb)

        temperature = 0.1  # <1 makes it sharper, >1 makes it flatter
        logits = torch.nn.functional.cosine_similarity(x, r, dim=-1) / temperature # (batch, n_tasks, n_rules)
        #logits = torch.sum(x * r, dim=-1) # Dot Product
        #logits = -torch.linalg.norm(x - r, dim=-1)  # Euclidean Distance: negative because smaller distance = better match

        return logits

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        batch_size = batch_c.shape[0]

        # === concept prediction ===
        c_embs = None
        c_probs, emb = self.encoder(batch_x) # batch, emb_size (denk ik?)
        c_pred = c_probs

        # === decode rules ===
        r = self.get_all_rule_vars() # tasks, total_rules, concepts, 3

        # === Calculate rule predictions ===
        logits_s = self.compute_rule_logits(emb)
        p_s = torch.softmax(logits_s, dim=-1) # batch, task, rules
        log_p_s = log_p_s = torch.log_softmax(logits_s, dim=-1)

        entr = torch.sum(batch_y*torch.sum(-p_s * torch.log(p_s + EPS), dim=-1)) / batch_y.shape[0]  # for logging

        # === calculate task prediction and 'concept reconstruction' ===
        pospolarity = r[:, :, :, 0]  # task, rule, concept
        irrelevance = r[:, :, :, 2]  # task, rule, concept
        relevance = 1-irrelevance  # task, rule, concept

        c_intv = batch_c.clone()
        batch_c = batch_c.unsqueeze(1).unsqueeze(1).repeat(1, self.n_tasks, self.effective_n_rules, 1)
        _pospolarity = pospolarity.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        _relevance = relevance.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        p_c_rec = self.rule_module.calc_c_rec(_pospolarity, _relevance)
        if self.training and not self.c_pred_in_logic:
            y_per_rule = self.rule_module.calc_y(batch_c, _pospolarity, _relevance)  # batch, task, rule, concept
        elif self.intervene:
            y_per_rule = self.rule_module.calc_y(batch_c, _pospolarity, _relevance)  # batch, task, rule, concept
            c_pred = c_intv
        else:  # use thresholded c_pred for y_pred
            y_per_rule = self.rule_module.calc_y((c_pred.detach() > 0.5).float().unsqueeze(1).unsqueeze(1).repeat(1, self.n_tasks, self.effective_n_rules, 1), _pospolarity, _relevance)

        return log_p_s, p_c_rec, y_per_rule, c_pred, p_s, entr

class CMR(MNISTModel):
    def __init__(self, 
                 encoder, 
                 emb_size, rule_emb_size, 
                 n_tasks, n_rules, n_concepts, 
                 concept_names,
                 learning_rate=0.001,
                 reset_selector=True, reset_selector_every_n_epochs=30,
                 rel_concept_counts=None, weight_concepts=False,
                 w_c=1, w_y=1, w_yF=1,
                 c_pred_in_logic=True, c_pred_in_rec=False,
                 orig_rule_sym_to_name=None):
        super().__init__(encoder, emb_size, rule_emb_size, n_tasks, n_rules, n_concepts, concept_names, ProbRDCat,
                         learning_rate, rel_concept_counts, weight_concepts, w_c, 10*w_y, w_yF, c_pred_in_logic, c_pred_in_rec,
                         orig_rule_sym_to_name, reset_selector, reset_selector_every_n_epochs, False, False)