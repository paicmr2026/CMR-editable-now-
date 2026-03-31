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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 1e-18

CONCEPT_EMB_SIZE = 16


#vragen
# class input types?
# training_step nog in log space
# onderste drie functies nodig?

class InputTypes: #DEZE KAN VERWIJDERD WORDEN?
    """
        Possible inputs for the rule selector
    """
    concepts = 0
    embedding = 1
    concepts_ground_truth = 2


class MNISTModel(pl.LightningModule): # IETS VAN RULESELECTOR.INPUT VERWIJDERD, VOOR VERSIMPELING
    def __init__(self, encoder, emb_size, rule_emb_size, n_tasks, n_rules, n_concepts, concept_names, rule_module,
                 lr=0.001,
                 rel_concept_counts=None, weight_concepts=False,
                 w_c=1, w_y=1, w_yF=1,
                 c_pred_in_logic=False,
                 c_pred_in_rec=False,
                 orig_rule_sym_to_name=None,
                 reset_selector=True, reset_selector_every_n_epochs=30,
                 intervene=False,
                 mutex=False):
        """
        Args:
            encoder: Torch module with forward method that returns (c_probs, emb)
            emb_size: Embedding size
            rule_emb_size: Rule embedding size
            n_tasks: Number of tasks
            n_rules: Allowed number of rules per task
            n_concepts: Number of concepts
            concept_names: Names of the concepts
            rule_module: RuleModule instance
            lr: Learning rate
            rel_concept_counts: Relative concept counts
            weight_concepts: Whether to weigh the concept reconstruction loss based on relative concept counts
            w_c: Weight for the concept reconstruction loss (w.r.t. the task loss)
            orig_rule_sym_to_name: None if printing rules should show pos polarity and irrelevance, otherwise prints pos and neg polarity
        """
        super().__init__()

        assert not mutex

        self.save_hyperparameters()

        self.reset_selector = reset_selector
        self.reset_selector_every_n_epochs = reset_selector_every_n_epochs

        self.lr = lr
        self.embedding_size = emb_size
        self.rule_emb_size = rule_emb_size
        self.n_tasks = n_tasks
        self.n_concepts = n_concepts
        self.n_rules = n_rules
        self.effective_n_rules = n_rules
        self.rel_concept_counts = rel_concept_counts
        self.weight_concepts = weight_concepts
        self.rule_logger = None
        self.concept_names = concept_names
        self.w_c = w_c
        self.w_y = w_y
        self.w_yF = w_yF
        self.orig_rule_sym_to_name = orig_rule_sym_to_name
        self.skip_info = False
        self.c_pred_in_logic = c_pred_in_logic
        self.c_pred_in_rec = c_pred_in_rec
        self.freeze_rules = False
        self.intervene = intervene

        self.encoder = encoder

        self.concept_embedder = ConceptEmbedding(in_features=self.embedding_size, n_concepts=self.n_concepts, emb_size=CONCEPT_EMB_SIZE)
        self.c_emb_combiner = torch.nn.Sequential(
            torch.nn.Linear(CONCEPT_EMB_SIZE * self.n_concepts, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size)
        )

        self.rule_module = rule_module(self.rule_emb_size, self.n_concepts, self.n_tasks, self.n_rules)

        self.info = defaultdict(list)
        self.val_info = defaultdict(list)

        # === For RuleAdd Interventions ===
        self.added_rules = []  # list of lists of rules

        self.rule_mask = torch.ones(self.n_tasks, self.n_rules)

        self.initialized = False
        self.initialize_rule_selector()

    def initialize_rule_selector(self):
        if self.initialized:
            def initialize_weights(module):
                if isinstance(module, torch.nn.Linear):
                    module.reset_parameters()
            self.neural_rule_selector.apply(initialize_weights)
            return
        self.initialized = True

        self.neural_rule_selector = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.n_tasks * self.effective_n_rules),
        ).to(self.device)

        if self.effective_n_rules > self.rule_mask.shape[1]:
            self.rule_mask = torch.cat([self.rule_mask, torch.ones(self.n_tasks, self.effective_n_rules - self.rule_mask.shape[1])], dim=-1)

        # === For Similarity Selector ===

        # Waarom doen we NN op embeddings van rules?
        self.rule_emb_proj = torch.nn.Sequential(
            torch.nn.Linear(self.rule_emb_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        )
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
        )

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        batch_size = batch_c.shape[0]

        # === concept prediction ===
        c_embs = None
        c_probs, emb = self.encoder(batch_x)
        c_pred = c_probs

        # === decode rules ===
        r = self.get_all_rule_vars() # tasks, total_rules, concepts, 3

        # === Calculate rule predictions ===
        logits_s = self.neural_rule_selector(emb).view(-1, self.n_tasks, self.effective_n_rules)
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

    def calc_avg_p_c_rec(self, dataloader): # AANGEPAST NAAR PROB SPACE
        self.train(True)
        noms = []
        dens = []

        for x in dataloader:
            batch_x, batch_c, batch_y = x
            if self.c_pred_in_rec:
                batch_c = self.predict_concepts(x).float()
            (_, p_c_rec, _, _, _, p_s, _) = self.forward(x)

            
            batch_y_btr = batch_y.unsqueeze(2).repeat(1, 1, self.effective_n_rules) # batch, task, rule
            batch_c_btrc = batch_c.unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1) # batch, task, rule, concept
            batch_y_btrc = batch_y_btr.unsqueeze(3).repeat(1, 1, 1, self.n_concepts) # batch, task, rule, concept

            # Weighted concept reconstruction: y * P(s) * P(c|rec)
            # batch, task, rule, concept
            weighted_p_c = batch_y_btrc * batch_y_btr.unsqueeze(-1) * p_s.unsqueeze(-1) * p_c_rec

            # Normalisation weight: y * P(s), summed over concepts axis is the same weight
            # batch, task, rule
            weighted_den = batch_y_btr * p_s

            noms.append(weighted_p_c)   # batch, task, rule, concept
            dens.append(weighted_den)   # batch, task, rule

        # Sum across the batch dimension (dim=0)
        nom = torch.cat(noms, dim=0).sum(dim=0)  # task, rule, concept
        den = torch.cat(dens, dim=0).sum(dim=0)  # task, rule

        # Normalise: average P(c | task, rule) weighted by y * P(s)
        avg_p_c_rec = nom / (den.unsqueeze(-1) + 1e-8)  # task, rule, concept

        return avg_p_c_rec

def training_step(self, batch, batch_idx): # ADJUSTED TO LOG SPACE
    batch_x, batch_c, batch_y = batch

    (p_s,        # batch, task, rule  (now direct probs, not log)
     p_c_rec,    # batch, task, rule, concept
     p_y,        # batch, task, rule
     p_c,        # batch, concept
     p_s_raw,
     entr,
     y_to_mask,
     ) = self.forward(batch)

    b_y_btr  = batch_y.unsqueeze(2).repeat(1, 1, self.effective_n_rules)
    b_c_btrc = batch_c.unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1)
    b_y_btrc = b_y_btr.unsqueeze(3).repeat(1, 1, 1, self.n_concepts)

    # Likelihoods in probability space (instead of log)
    p_y_true     = torch.exp(-binary_cross_entropy(p_y,    b_y_btr,  reduction='none'))  # batch, task, rule
    p_c_true     = torch.exp(-binary_cross_entropy(p_c,    batch_c,  reduction='none'))  # batch, concept

    if not self.c_pred_in_rec:
        p_c_rec_true = torch.exp(-binary_cross_entropy(p_c_rec, b_c_btrc, reduction='none'))
    else:
        c_pred_btrc  = (p_c.detach() > 0.5).float().unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1)
        p_c_rec_true = torch.exp(-binary_cross_entropy(p_c_rec, c_pred_btrc, reduction='none'))

    # Concept reconstruction term
    if not self.weight_concepts:
        sum1 = torch.prod(p_c_rec_true ** b_y_btrc, dim=-1)   # batch, task, rule
    else:
        concept_w   = 1 / self.rel_concept_counts.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_y.shape[0], self.n_tasks, self.effective_n_rules, 1).to(DEVICE)
        i_concept_w = 1 / (1 - self.rel_concept_counts.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_y.shape[0], self.n_tasks, self.effective_n_rules, 1)).to(DEVICE)
        w    = torch.where(b_c_btrc == 1, concept_w, i_concept_w)
        sum1 = torch.prod((p_c_rec_true ** b_y_btrc) ** (self.w_c * w), dim=-1)

    # Per-label y weighting
    w2 = torch.ones_like(batch_y.unsqueeze(-1).repeat(1, 1, self.effective_n_rules))
    w2[batch_y == 0] = self.w_yF

    # Combine rule contributions  —  multiply instead of logsumexp
    temp = p_s * (p_y_true ** (0.1 * self.w_y * w2)) * (sum1 ** self.w_c)  # batch, task, rule
    temp = torch.sum(temp, dim=-1)                                            # batch, task  (marginalise over rules)

    # Probability per sample
    prob_per_sample = torch.prod(temp, dim=-1)                        # batch
    prob_per_sample = prob_per_sample * torch.prod(p_c_true, dim=-1)  # multiply concept likelihood

    loss = torch.mean(-torch.log(prob_per_sample + 1e-8))  # back to NLL for the loss scalar

    # === information ===
    if self.skip_info:
        return loss

    y_pred = torch.einsum("btr,btr->bt", p_s[y_to_mask[:, 0, 0] == 1], p_y[y_to_mask[:, 0, 0] == 1])
    avg_c_per_task = torch.einsum('br,brc->bc',
                                  p_s[torch.logical_and(batch_y == 1, y_to_mask[:, :, 0] == 1)],
                                  p_c_rec[torch.logical_and(batch_y == 1, y_to_mask[:, :, 0] == 1)])

    c_true     = batch_c.unsqueeze(1).repeat(1, self.n_tasks, 1)
    c_true     = c_true[torch.logical_and(batch_y == 1, y_to_mask[:, :, 0] == 1)]
    c_true     = c_true.view(-1, self.n_concepts).detach().cpu().numpy()
    c_avg_pred = avg_c_per_task.detach().cpu().reshape(-1, self.n_concepts).numpy().round()
    c_true, c_avg_pred = c_true.flatten(), c_avg_pred.flatten()

    c_avg_accuracy  = accuracy_score(c_true, c_avg_pred)
    c_accuracy      = accuracy_score(batch_c.cpu().numpy().flatten(), p_c.detach().cpu().numpy().round().flatten())
    y_pred          = y_pred > 0.5
    y_accuracy_logic = accuracy_score(batch_y.detach().cpu()[y_to_mask[:, 0, 0].to('cpu') == 1], y_pred.detach().cpu())

    self.info["c_accuracy"].append(c_accuracy)
    self.info["c'_accuracy"].append(c_avg_accuracy)
    self.info["y_accuracy"].append(y_accuracy_logic)
    self.info["entropy"].append(entr.item())
    self.info["loss"].append(loss.item())
    self.info["p_s"].append(p_s.detach())

    return loss
    


## DEZE ONDERSTE DRIE ZIJN VOLGENS MIJ NIET NUTTIG? (of wel?...)


    def get_rules_sym(self, rule_vars, rule_idx=None, task_idx=None):
        def to_rule_sym(r_idx, t_idx, rule_vars):
            c_type = torch.argmax(rule_vars[t_idx, r_idx, :, :].detach(), dim=-1)
            f = lambda argmax: 1 if argmax == 0 else 0 if argmax == 1 else 9
            r = [f(c_type[k]) for k in range(len(c_type))]
            if self.orig_rule_sym_to_name is None:  # show pospol and (irrel) only
                r = [f'({self.concept_names[k]})' if r[k] == 9 else self.concept_names[k] for k in range(len(r)) if r[k] in (1, 9)]
            else:  # show pospol and ~negpol only
                r = [f'~{self.concept_names[k]}' if r[k] == 0 else self.concept_names[k] for k in range(len(r)) if r[k] in (1, 0)]
            rule_sym = ' & '.join(r)  # only showing positive polarity atoms and not relevant atoms
            return rule_sym
        if rule_idx is not None and task_idx is not None:
            return to_rule_sym(rule_idx, task_idx, rule_vars)
        elif rule_idx is not None:
            return [to_rule_sym(rule_idx, t, rule_vars) for t in range(self.n_tasks)]
        elif task_idx is not None:
            return [to_rule_sym(r, task_idx, rule_vars) for r in range(self.effective_n_rules)]
        else:  # all of them
            return [[to_rule_sym(r, t, rule_vars) for r in range(self.effective_n_rules)] for t in range(self.n_tasks)]

    def aggregate_rules(self, dataloader, type='most_likely', inv=False):
        assert type in ('most_likely', 'mean_probability', 'concept_probs')
        rule_vars = self.get_all_rule_vars()
        rules_sym = self.get_rules_sym(rule_vars)
        if type == 'most_likely':
            task_to_rules = dict()
            task_to_rule_idx = dict()
            for task in range(self.n_tasks):
                task_to_rules[task] = dict()
                task_to_rule_idx[task] = set()
            for batch in dataloader:
                log_p_s_x, _, y_per_rule, _, _, _, _ = self(batch)
                rule_idxs = torch.argmax(log_p_s_x, dim=-1)
                for example_idx, y_true in enumerate(batch[2]):
                    for task in range(self.n_tasks):
                        if not y_true[task] and not inv:
                            continue
                        if y_true[task] and inv:
                            continue
                        rule_idx = rule_idxs[example_idx, task]
                        rule_sym = rules_sym[task][rule_idx]
                        if rule_sym not in task_to_rules[task]:
                            task_to_rules[task][rule_sym] = 0
                        task_to_rules[task][rule_sym] += 1
                        task_to_rule_idx[task].add(rule_idx.item())
            return task_to_rules, task_to_rule_idx
        elif type == 'mean_probability':
            rule_probs_per_task_T = [[[] for _ in range(self.effective_n_rules)] for _ in range(self.n_tasks)]  # n_tasks x n_rules x ..
            rule_probs_per_task_F = [[[] for _ in range(self.effective_n_rules)] for _ in range(self.n_tasks)]  # n_tasks x n_rules x ..
            for batch in dataloader:
                log_p_s_x, _, y_per_rule, _, _, _, _ = self(batch)
                for example_idx, y_true in enumerate(batch[2]):
                    for task in range(self.n_tasks):
                        arr = rule_probs_per_task_T if y_true[task] else rule_probs_per_task_F
                        for rule_idx, rule_prob in enumerate(torch.exp(log_p_s_x[example_idx, task]).detach()):
                            arr[task][rule_idx].append(rule_prob.item())
            rule_probs_per_task_T, rule_probs_per_task_F = (list(map(torch.tensor, rule_probs_per_task_T)), list(map(torch.tensor, rule_probs_per_task_F)))
            rule_probs_per_task_T_avg, rule_probs_per_task_F_avg = [x.mean(dim=-1) for x in rule_probs_per_task_T], [x.mean(dim=-1) for x in rule_probs_per_task_F]
            rule_probs_per_task_T_sdev, rule_probs_per_task_F_sdev = [x.std(dim=-1) for x in rule_probs_per_task_T], [x.std(dim=-1) for x in rule_probs_per_task_F]
            return rule_probs_per_task_T_avg, rule_probs_per_task_F_avg, rule_probs_per_task_T_sdev, rule_probs_per_task_F_sdev
        elif type == 'concept_probs':
            concept_probs = torch.zeros((self.n_tasks, self.effective_n_rules, self.n_concepts, 2))  # task x rule x concept x 2
            for batch in dataloader:
                b_x, b_c, b_y = batch
                b_c1 = b_c.unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1)
                b_y1 = b_y.unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.effective_n_rules, self.n_concepts)
                _, _, _, _, p_s, _, _ = self(batch)
                for b_idx in range(p_s.shape[0]):
                    probs = p_s[b_idx, :, :].unsqueeze(-1).repeat(1, 1, self.n_concepts)
                    concept_probs[:, :, :, 0] += probs * b_c1[b_idx, :, :, :].float() * b_y1[b_idx, :, :, :].float()
                    concept_probs[:, :, :, 1] += probs * (1-b_c1[b_idx, :, :, :]).float() * b_y1[b_idx, :, :, :].float()
            concept_probs /= len(dataloader.dataset)
            return concept_probs

    def check_polarity_crispness(self, used_idxsT, used_idxsF):
        rule_vars = self.get_all_rule_vars()  # task, rule, concept, 3
        temp = torch.all(rule_vars < 0.9, dim=-1).float()  # task, rule, concept
        for t_idx in range(temp.shape[0]):
            for r_idx in range(temp.shape[1]):
                if r_idx not in used_idxsT[t_idx] or r_idx not in used_idxsF[t_idx]:
                    continue
                rule_sym = self.get_rules_sym(rule_vars, r_idx, t_idx)
                if temp[t_idx, r_idx].sum().item() > 0.01:
                    print("Rule", rule_sym, "for task", t_idx, "does not have crisp polarity!")
                    for c_idx in range(temp.shape[1]):
                        if temp[t_idx, r_idx, c_idx].item() > 0.01:
                            print("-", c_idx, ":", rule_vars[t_idx, r_idx, c_idx])

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