import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from model_simmilarity import (
    MNISTModel, MNISTEncoder, ProbRDCat,
    InputTypes, SaveBestModelCallbackVal,
    get_accuracy, get_concept_accuracy
)
from experiments.mnist.mnist_dataset import addition_dataset, create_single_digit_addition

# ── Config ───────────────────────────────────────────
NUM_DIGITS   = 2
DIGIT_LIMIT  = 10
EMB_SIZE     = 500
RULE_EMB     = 1000
N_RULES      = 20
LR           = 0.0001
BATCH_SIZE   = 512
MAX_EPOCHS   = 100
VAL_SPLIT    = 0.1
SEED         = 42

def c_idx_to_name(c_idx, num_digits, digit_limit):
    d = c_idx // digit_limit
    v = c_idx % digit_limit
    return f"d{d}_{v}"

def main():
    pl.seed_everything(SEED)

    # ── Dataset ───────────────────────────────────────────────────────────────
    x_train_raw, c_train_raw, y_train_raw = addition_dataset(True,  NUM_DIGITS, DIGIT_LIMIT)
    x_test_raw,  c_test_raw,  y_test_raw  = addition_dataset(False, NUM_DIGITS, DIGIT_LIMIT)

    x_train = torch.stack(x_train_raw, dim=1).float()
    c_train = torch.cat(c_train_raw, dim=-1).float()
    y_train = F.one_hot(y_train_raw.unsqueeze(-1).long().ravel()).float()

    x_test = torch.stack(x_test_raw, dim=1).float()
    c_test = torch.cat(c_test_raw, dim=-1).float()
    y_test = F.one_hot(y_test_raw.unsqueeze(-1).long().ravel()).float()

    # Train / Val split
    split = int(len(x_train) * (1 - VAL_SPLIT))
    x_val,  c_val,  y_val  = x_train[split:], c_train[split:], y_train[split:]
    x_train, c_train, y_train = x_train[:split], c_train[:split], y_train[:split]

    train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_val,   c_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TensorDataset(x_test,  c_test,  y_test),  batch_size=BATCH_SIZE)

    n_concepts = c_train.shape[1]
    n_tasks    = y_train.shape[1]

    # ── Model ────────────────────────────────────────
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
        reset_selector=True,
        reset_selector_every_n_epochs=40,
        w_c=1, w_y=1, w_yF=1,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    cb = SaveBestModelCallbackVal()
    checkpoint = ModelCheckpoint(
        dirpath="./results/mnist_base/CMR/",
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[cb, checkpoint],
        deterministic=True,
        gradient_clip_val=0.5,        # Caps the norm of the gradients
        gradient_clip_algorithm="norm" # Standard approach for stability
    )
    trainer.fit(model, train_loader, val_loader)

    # Herstellen van de beste gewichten
    model.load_state_dict(cb.best_state_dict)
    model.eval()
    model.to("cpu")

    # ── Evaluatie ─────────────────────────────────────────────────────────────
    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)
    
    print(f"\nTrain Acc: {train_acc:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    # Rules
    task_to_rules, _ = model.aggregate_rules(train_loader, type='most_likely')
    for task in sorted(task_to_rules.keys()):
        print(f"Task {task}: {task_to_rules[task]}")

if __name__ == "__main__":
    main()