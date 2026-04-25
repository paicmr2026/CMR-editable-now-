import torch
torch.cuda.empty_cache()
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

# Importing components as seen in your AECMR_train.py and local_experiments.py
from experiments.mnist.autoencoderCMR import (
    MNISTModel, MNISTEncoder, AECat,
    InputTypes, SimilarityTypes, SaveBestModelCallbackVal,
    get_accuracy, get_concept_accuracy
)
# Using CUB architecture and metadata
from experiments.cub.models import CUBEncoder
from experiments.cub.CUB200.cub_loader import CLASS_NAMES, CONCEPT_SEMANTICS # voor Rules beter te kunnen tonen

# ── CUB-Specific Config ───────────────────────────────────────────
EMB_SIZE     = 100   # Matching your cub_cmr settings in local_experiments.py
RULE_EMB     = 500   # Matching local_experiments.py
N_RULES      = 3     # Matching local_experiments.py
LR           = 0.001
BATCH_SIZE   = 1280
MAX_EPOCHS   = 300
SEED         = 42

def main():
    pl.seed_everything(SEED)

    # ── Dataset Loading (Using saved embeddings) ─────────────────────
    # Path based on save_embeddings.py output
    emb_dir = '/embeddings/cub/'
    
    # Load processed embeddings
    x_train, c_train, y_train = torch.load(os.path.join(emb_dir, 'train_embeddings.pt'))
    x_test, c_test, y_test = torch.load(os.path.join(emb_dir, 'test_embeddings.pt'))

    train_loader = DataLoader(
        TensorDataset(x_train, c_train.float(), y_train.float()), 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, c_test.float(), y_test.float()), 
        batch_size=BATCH_SIZE
    )

    # ── Model Initialization ────────────────────────────────────────
    # Replacing MNISTEncoder with CUBEncoder
    model = MNISTModel(
        encoder=CUBEncoder(
            input_size=x_train.shape[1], 
            emb_size=EMB_SIZE, 
            cp_output=c_train.shape[1]
        ),
        rule_module=AECat, # Following AECMR_train.py's use of Autoencoder Category
        n_concepts=c_train.shape[1],
        n_tasks=y_train.shape[1],
        n_rules=N_RULES,
        rule_emb_size=RULE_EMB,
        emb_size=EMB_SIZE,
        selector_input=InputTypes.embedding,
        lr=LR,
        w_c=1, 
        w_y=30, 
        w_yF=0.005,
        reset_selector=True,
        reset_selector_every_n_epochs=25
    )

    # ── Training ───────────────────────────────────────────────────
    cb = SaveBestModelCallbackVal()
    checkpoint = ModelCheckpoint(
        dirpath="./results/cub_base/CMR/",
        filename="best_cub_cmr",
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[cb, checkpoint],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        deterministic=True
    )
    
    trainer.fit(model, train_loader, train_loader)

    # ── Evaluation ─────────────────────────────────────────────────
    model.load_state_dict(cb.best_state_dict)
    model.eval()

    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)
    
    print(f"\nCUB Train Acc: {train_acc:.4f}")
    print(f"CUB Test Acc:  {test_acc:.4f}")

    # Generate Interpretability Rules
    task_to_rules, _ = model.aggregate_rules(train_loader, type='most_likely')
    for task_idx, rules in task_to_rules.items():
        print(f"=== Species: {CLASS_NAMES[task_idx]} ===")
        for rule in rules:
            print(f"  Rule: {rule}")

if __name__ == '__main__':
    main()