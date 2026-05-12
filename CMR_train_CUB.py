import torch
torch.cuda.empty_cache()
import os
import pandas as pd
import lightning.pytorch as pl

from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint

from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from experiments.mnist.models_copy import (
    MNISTModel, MNISTEncoder, \
    InputTypes, SaveBestModelCallbackVal, ProbRDCat, get_accuracy, get_concept_accuracy
)
from experiments.cub.models import CUBEncoder
from experiments.cub.CUB200.cub_loader import CONCEPT_SEMANTICS, CLASS_NAMES

# --- CONFIGURATION FLAGS ---
RETRAIN      = False
CHECKPOINT_PATH = "./results/cub_base/CMR/best_cub_cmr.ckpt"
# ---------------------------

EMB_SIZE     = 100   
RULE_EMB     = 500   
N_RULES      = 3     
LR           = 0.001
BATCH_SIZE   = 100
MAX_EPOCHS   = 250
SEED         = 42

def main():
    pl.seed_everything(SEED)

    root = './experiments/cub/embeddings/cub/'
    
    x_train = torch.load(os.path.join(root, 'train_x.pt')).float()
    c_train = torch.load(os.path.join(root, 'train_c.pt')).float()
    y_train = torch.load(os.path.join(root, 'train_y.pt'))
    y_train = torch.nn.functional.one_hot(y_train.long(), num_classes=200).float()

    x_test = torch.load(os.path.join(root, 'test_x.pt')).float()
    c_test = torch.load(os.path.join(root, 'test_c.pt')).float()
    y_test = torch.load(os.path.join(root, 'test_y.pt'))
    y_test = torch.nn.functional.one_hot(y_test.long(), num_classes=200).float()

    train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=BATCH_SIZE)

    model = MNISTModel(
        encoder=CUBEncoder(
            input_size=x_train.shape[1], 
            emb_size=EMB_SIZE, 
            cp_output=c_train.shape[1]
        ),
        emb_size=EMB_SIZE,
        rule_emb_size=RULE_EMB,
        n_concepts=c_train.shape[1],
        n_tasks=y_train.shape[1],
        n_rules=N_RULES,
        concept_names=CONCEPT_SEMANTICS,
        rule_module=ProbRDCat,
        lr=LR,
        w_c=1, 
        w_y=30, 
        w_yF=0.005,
        reset_selector=True,
        reset_selector_every_n_epochs=30
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.dirname(CHECKPOINT_PATH),
        filename=os.path.basename(CHECKPOINT_PATH).replace('.ckpt', ''),
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        deterministic=True
    )
    
    if RETRAIN:
        print("--- Starting Training Phase ---")
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=train_loader)
        best_model_path = checkpoint_callback.best_model_path
    else:
        print(f"--- Testing Mode: Loading weights from {CHECKPOINT_PATH} ---")
        best_model_path = CHECKPOINT_PATH

    model = MNISTModel.load_from_checkpoint(
        best_model_path,
        encoder=CUBEncoder(input_size=x_train.shape[1], emb_size=EMB_SIZE, cp_output=c_train.shape[1]),
        concept_names=CONCEPT_SEMANTICS,
        weights_only=False
    )

    model.eval()
    model.to("cpu")

    print("\n--- Final Evaluations ---")
    train_c_acc = get_concept_accuracy(model, train_loader)
    test_c_acc  = get_concept_accuracy(model, test_loader)

    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)

    print(f"\nTrain c accuracy:  {train_c_acc:.4f}")
    print(f"Test c accuracy:   {test_c_acc:.4f}")

    print(f"\nTrain accuracy:  {train_acc:.4f}")
    print(f"Test accuracy:   {test_acc:.4f}")

    ## Rules bekijken
    print("\nExtracting learned rules...")
    rules = []
    task_to_rules, _ = model.aggregate_rules(test_loader, type='most_likely')

    print("\n" + "="*50)
    print("LEARNED CONCEPT RULES (Sample)")
    print("="*50)

    for task in task_to_rules:
                    print('=== Task', CLASS_NAMES[task], "= True ", "===")
                    for rule in task_to_rules[task]:
                        print(rule, ":", task_to_rules[task][rule])
                        rule_latex = rule.replace('&', '\land').replace('|', '\lor').replace('~', '\\neg ')
                        rules.append([CLASS_NAMES[task], rule_latex, task_to_rules[task][rule]])

if __name__ == '__main__':
    main()