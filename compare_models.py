import torch
import os
import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch.nn.functional as F
from torch import nn
import csv
from torch.utils.data import DataLoader, TensorDataset

from experiments.mnist.autoencoderCMR import (
    MNISTModel as AECMRModel, MNISTEncoder, AECat,
    InputTypes, SimilarityTypes, SaveBestModelCallbackVal,
    get_accuracy
)
from experiments.mnist.mnist_dataset import addition_dataset
from experiments.mnist.models_copy import (
    MNISTModel as CMRModel, ProbRDCat
)
from experiments.cub.models import CUBEncoder
from experiments.cub.CUB200.cub_loader import CONCEPT_SEMANTICS

'''
Neural Net (DNN)
'''
from sklearn.metrics import accuracy_score
from torch.nn.functional import binary_cross_entropy

class DNN(pl.LightningModule):
    def __init__(self, input_size, emb_size, n_tasks, n_hidden_layers=4, lr=0.001, is_mnist=False, dropout=0.5):
        super().__init__()
        self.lr = lr
        self.is_mnist = is_mnist
        
        if is_mnist:
            self.encoder = MNISTEncoder(emb_size, 10, 2)
            input_dim = emb_size
        else:
            self.encoder = None
            input_dim = input_size

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_dim, emb_size))
        self.layers.append(torch.nn.BatchNorm1d(emb_size))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Dropout(dropout))

        for _ in range(n_hidden_layers):
            self.layers.append(torch.nn.Linear(emb_size, emb_size))
            self.layers.append(torch.nn.BatchNorm1d(emb_size))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(dropout))

        self.layers.append(torch.nn.Linear(emb_size, n_tasks))
        self.training_step_outputs = []

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x = x[0]
        if self.is_mnist:
            x = self.encoder(x)
            if isinstance(x, (tuple, list)):
                x = x[1] if len(x) > 1 else x[0]
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        _, _, batch_y = batch
        logits = self.forward(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y)
        self.training_step_outputs.append({"preds": logits.detach(), "targets": batch_y.detach()})
        return loss

    def on_train_epoch_end(self):
        if not self.training_step_outputs: return
        all_logits = torch.cat([x["preds"] for x in self.training_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.training_step_outputs])
        acc = (torch.argmax(all_logits, dim=1) == torch.argmax(all_targets, dim=1)).float().mean().item()
        print(f"Epoch {self.current_epoch} - Training Acc: {acc:.4f}")
        self.training_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

def accuracy_DNN(train_loader, val_loader, test_loader, params):
    pl.seed_everything(params['seed'])
    model = DNN(
        input_size=params['input_size'],
        emb_size=params['emb_size'],
        n_tasks=params['n_tasks'],
        n_hidden_layers=params['n_hidden_layers'],
        lr=params['lr'],
        is_mnist=params.get('is_mnist', False),
        dropout=0.5 if not params.get('is_mnist') else 0.2
    )
    model = run_training(model, train_loader, val_loader, params['epochs'])
    model.eval()
    acc, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            _, _, y = batch
            logits = model(batch)
            acc += (torch.argmax(logits, dim=1) == torch.argmax(y, dim=1)).sum().item()
            total += y.size(0)
    return acc / total

def run_training(model, train_loader, val_loader, max_epochs):
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        deterministic=True,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        enable_checkpointing=False,
        logger=False
    )
    trainer.fit(model, train_loader, val_loader)
    return model

def get_cub_data():
    root = './experiments/cub/embeddings/cub/'
    x_train = torch.load(os.path.join(root, 'train_x.pt')).float()
    c_train = torch.load(os.path.join(root, 'train_c.pt')).float()
    y_train = torch.nn.functional.one_hot(torch.load(os.path.join(root, 'train_y.pt')).long(), num_classes=200).float()
    x_test = torch.load(os.path.join(root, 'test_x.pt')).float()
    c_test = torch.load(os.path.join(root, 'test_c.pt')).float()
    y_test = torch.nn.functional.one_hot(torch.load(os.path.join(root, 'test_y.pt')).long(), num_classes=200).float()
    return x_train, c_train, y_train, x_test, c_test, y_test

def get_mnist_data():
    x_tr_raw, c_tr_raw, y_tr_raw = addition_dataset(True, 2, 10)
    x_te_raw, c_te_raw, y_te_raw = addition_dataset(False, 2, 10)
    
    x_train = torch.stack(x_tr_raw, dim=1).float()
    c_train = torch.cat(c_tr_raw, dim=-1).float()
    y_train = F.one_hot(y_tr_raw.unsqueeze(-1).long().ravel()).float()
    
    x_test = torch.stack(x_te_raw, dim=1).float()
    c_test = torch.cat(c_te_raw, dim=-1).float()
    y_test = F.one_hot(y_te_raw.unsqueeze(-1).long().ravel()).float()
    return x_train, c_train, y_train, x_test, c_test, y_test

def accuracy_Majority(train_loader, test_loader):
    all_y_train = []
    for _, _, y in train_loader:
        all_y_train.append(torch.argmax(y, dim=1))
    all_y_train = torch.cat(all_y_train)
    majority_class = torch.mode(all_y_train).values.item()
    
    acc, total = 0, 0
    for _, _, y in test_loader:
        test_labels = torch.argmax(y, dim=1)
        acc += (test_labels == majority_class).sum().item()
        total += y.size(0)
    return acc / total

def accuracy_CMR(train_loader, val_loader, test_loader, config, is_cub=True):
    pl.seed_everything(config['seed'])
    encoder = CUBEncoder(config['x_dim'], config['emb'], config['c_dim']) if is_cub else MNISTEncoder(config['emb'], 10, 2)
    
    model = CMRModel(
        encoder=encoder, 
        emb_size=config['emb'], 
        rule_emb_size=config['remb'],
        n_concepts=config['c_dim'], 
        n_tasks=config['y_dim'], 
        n_rules=config['rules'],
        concept_names=config['c_names'], 
        rule_module=ProbRDCat, 
        lr=config['lr'],
        w_c=config.get('w_c', 1),
        w_y=config.get('w_y', 1),
        w_yF=config.get('w_yF', 1)
    )
    model = run_training(model, train_loader, val_loader, config['epochs'])
    return get_accuracy(model, test_loader)

def accuracy_AECMR(train_loader, val_loader, test_loader, config, is_cub=True):
    pl.seed_everything(config['seed'])
    encoder = CUBEncoder(config['x_dim'], config['emb'], config['c_dim']) if is_cub else MNISTEncoder(config['emb'], 10, 2)
    
    model = AECMRModel(
        encoder=encoder, 
        emb_size=config['emb'], 
        rule_emb_size=config['remb'],
        n_concepts=config['c_dim'], 
        n_tasks=config['y_dim'], 
        n_rules=config['rules'],
        concept_names=config['c_names'], 
        rule_module=AECat, 
        lr=config['lr'],
        w_c=config.get('w_c', 1),
        w_y=config.get('w_y', 1),
        w_yF=config.get('w_yF', 1),
        w_embedding_ae_loss=config.get('w_embedding_ae_loss', 0.5),
        w_rule_ae_loss=config.get('w_rule_ae_loss', 8),
        temperature=0.1
    )
    model = run_training(model, train_loader, val_loader, config['epochs'])
    return get_accuracy(model, test_loader)

def main():
    EPOCHS_MNIST = 50
    EPOCHS_CUB = 100
    SEEDS = 10
    BATCH_SIZE = 2048
    NUM_WORKERS = 15
    REPLACE_CSV = False
    
    do_CMR      = False
    do_AECMR    = False
    do_DNN      = True
    do_Majority = False

    csv_dir = './results'
    csv_path = os.path.join(csv_dir, 'model_comparison_results.csv')
    os.makedirs(csv_dir, exist_ok=True)

    header = ['Dataset', 'Model', 'Seed', 'Accuracy']
    if REPLACE_CSV or not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    datasets = ['CUB']#['MNIST', 'CUB']
    models = ['CMR', 'AECMR', 'DNN', 'Majority']
    results = {d: {m: [] for m in models} for d in datasets}

    for ds in datasets:
        if ds == 'MNIST':
            x_tr, c_tr, y_tr, x_te, c_te, y_te = get_mnist_data()
            config = {
                'x_dim': x_tr.shape[1], 
                'c_dim': c_tr.shape[1], 
                'y_dim': y_tr.shape[1], 
                'emb': 500, 
                'remb': 1000, 
                'rules': 20, 
                'lr': 0.0001, 
                'epochs': EPOCHS_MNIST, 
                'c_names': [f"d{i//10}_{i%10}" for i in range(20)],
                'w_c': 1, 
                'w_y': 1, 
                'w_yF': 1,
                'w_embedding_ae_loss': 0.5,
                'w_rule_ae_loss': 8
            }
        else:
            x_tr, c_tr, y_tr, x_te, c_te, y_te = get_cub_data()
            config = {
                'x_dim': x_tr.shape[1], 
                'c_dim': c_tr.shape[1], 
                'y_dim': y_tr.shape[1], 
                'emb': 100, 
                'remb': 500, 
                'rules': 3, 
                'lr': 0.001, 
                'epochs': EPOCHS_CUB, 
                'c_names': CONCEPT_SEMANTICS,
                'w_c': 1, 
                'w_y': 30, 
                'w_yF': 0.005,
                'temperature': 0.1,
                'reset_selector': True,
                'reset_selector_every_n_epochs': 40
            }

        train_ds = TensorDataset(x_tr, c_tr, y_tr)
        test_ds = TensorDataset(x_te, c_te, y_te)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True)

        for seed in range(SEEDS):
            config['seed'] = seed
            print(f"Running {ds} - Seed {seed}")
            
            run_configs = []
            if do_CMR: run_configs.append(('CMR', lambda: accuracy_CMR(train_loader, train_loader, test_loader, config, ds=='CUB')))
            if do_AECMR: run_configs.append(('AECMR', lambda: accuracy_AECMR(train_loader, train_loader, test_loader, config, ds=='CUB')))
            if do_DNN:
                if ds == 'MNIST':
                    d_params = {
                        'input_size': 0, 
                        'emb_size': 100, 
                        'n_tasks': y_tr.shape[1], 
                        'n_hidden_layers': 2, 
                        'lr': 0.001, 
                        'epochs': config['epochs'], 
                        'seed': seed,
                        'is_mnist': True 
                    }
                else:
                    d_params = {
                        'input_size': x_tr.shape[1], 
                        'emb_size': 500, 
                        'n_tasks': y_tr.shape[1], 
                        'n_hidden_layers': 2, 
                        'lr': 0.01, 
                        'epochs': config['epochs'], 
                        'seed': seed,
                        'is_mnist': False
                    }
                run_configs.append(('DNN', lambda: accuracy_DNN(train_loader, train_loader, test_loader, d_params)))
            if do_Majority: run_configs.append(('Majority', lambda: accuracy_Majority(train_loader, test_loader)))

            for m_name, m_func in run_configs:
                acc = m_func()
                results[ds][m_name].append(acc)
                with open(csv_path, mode='a', newline='') as f:
                    csv.writer(f).writerow([ds, m_name, seed, acc])

    for ds in datasets:
        print(f"\n--- Results for {ds} ---")
        for m in models:
            accs = results[ds][m]
            if accs:
                print(f"{m}: Mean Acc = {np.mean(accs):.4f}, Std = {np.std(accs):.4f}")

if __name__ == "__main__":
    main()