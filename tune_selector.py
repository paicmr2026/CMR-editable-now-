import os
import torch
import csv
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint

from model_similarity import (
    MNISTModel, MNISTEncoder, ProbRDCat,
    InputTypes, SimilarityTypes, SaveBestModelCallbackVal,
    get_accuracy
)
from experiments.mnist.mnist_dataset import addition_dataset

# ── Config ───────────────────────────────────────────
NUM_DIGITS   = 2
DIGIT_LIMIT  = 10
EMB_SIZE     = 500
RULE_EMB     = 1000
N_RULES      = 20
LR           = 0.0001
BATCH_SIZE   = 256
MAX_EPOCHS   = 30
VAL_SPLIT    = 0.1
SEED         = 42

def c_idx_to_name(c_idx, num_digits, digit_limit):
    d = c_idx // digit_limit
    v = c_idx % digit_limit
    return f"d{d}_{v}"

def main():
    pl.seed_everything(SEED)
    results = []

    # ── Dataset Setup ──────────────────────────────────
    x_train_raw, c_train_raw, y_train_raw = addition_dataset(True,  NUM_DIGITS, DIGIT_LIMIT)
    x_test_raw,  c_test_raw,  y_test_raw  = addition_dataset(False, NUM_DIGITS, DIGIT_LIMIT)

    x_train = torch.stack(x_train_raw, dim=1).float()
    c_train = torch.cat(c_train_raw, dim=-1).float()
    y_train = F.one_hot(y_train_raw.unsqueeze(-1).long().ravel()).float()

    x_test = torch.stack(x_test_raw, dim=1).float()
    c_test = torch.cat(c_test_raw, dim=-1).float()
    y_test = F.one_hot(y_test_raw.unsqueeze(-1).long().ravel()).float()

    split = int(len(x_train) * (1 - VAL_SPLIT))
    train_loader = DataLoader(TensorDataset(x_train[:split], c_train[:split], y_train[:split]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_train[split:], c_train[split:], y_train[split:]), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=BATCH_SIZE)

    n_concepts = c_train.shape[1]
    n_tasks    = y_train.shape[1]

    # ── Grid Search Configurations ─────────────────────
    # small=1, medium=3, large=5
    configs = [
        #{"layers": 1, "bn": False, "desc": "small_no_bn"},
        #{"layers": 1, "bn": True,  "desc": "small_bn"},
        #{"layers": 3, "bn": False, "desc": "medium_no_bn"},
        #{"layers": 3, "bn": True,  "desc": "medium_bn"},
        #{"layers": 2, "bn": False, "desc": "2_no_bn"},
        #{"layers": 3, "bn": False, "desc": "3_no_bn"},
        #{"layers": 4, "bn": False, "desc": "4_no_bn"},
        #{"layers": 5, "bn": False, "desc": "5_no_bn"},
        #{"layers": 6, "bn": False, "desc": "6_no_bn"} 
        {"sim": SimilarityTypes.cosine, "desc": "Cosine"},
        {"sim": SimilarityTypes.dotproduct, "desc": "Dotproduct"},
        {"sim": SimilarityTypes.euclidean, "desc": "Euclidean"},
        {"sim": SimilarityTypes.bilinear, "desc": "Bilinear"},
        {"sim": SimilarityTypes.attention, "desc": "Attention"}
    ]

    for conf in configs:
        print(f"\n>>> TESTING CONFIGURATION: {conf['desc']}")
        
        model = MNISTModel(
            encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
            emb_size=EMB_SIZE,
            rule_emb_size=RULE_EMB,
            n_tasks=n_tasks,
            n_rules=N_RULES,
            n_concepts=n_concepts,
            concept_names=[c_idx_to_name(i, NUM_DIGITS, DIGIT_LIMIT) for i in range(n_concepts)],
            rule_module=ProbRDCat,
            lr=LR,
            selector_input=InputTypes.embedding,
            selector_similarity=conf['sim'],
            reset_selector=False,
            reset_selector_every_n_epochs=30,
            temperature=0.1,
            hidden_layers=3,
            use_batch_norm=False
        )

        cb = SaveBestModelCallbackVal()
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=[cb],
            deterministic=True,
            enable_checkpointing=False,
            logger=False
        )
        
        trainer.fit(model, train_loader, val_loader)

        # Evaluate best model
        model.load_state_dict(cb.best_state_dict)
        model.eval()
        
        train_acc = get_accuracy(model, train_loader)
        test_acc  = get_accuracy(model, test_loader)

        results.append({
            "config": conf['desc'],
            "train_acc": train_acc,
            "test_acc": test_acc,
            "best_val_loss": cb.best_loss.item() if torch.is_tensor(cb.best_loss) else cb.best_loss
        })

    # ── Save Results ───────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv("tune_selector_results.csv", index=False)
    print("\nTesting complete. Results saved to tune_selector_results.csv")

if __name__ == "__main__":
    main()