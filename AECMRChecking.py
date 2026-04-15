import torch
import torch.nn as nn
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from autoencoderCMR import (
    MNISTModel, MNISTEncoder, AECat,
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
MAX_EPOCHS   = 500
VAL_SPLIT    = 0.1
SEED         = 42
VERBOSE = False

def c_idx_to_name(c_idx, num_digits, digit_limit):
    d = c_idx // digit_limit
    v = c_idx % digit_limit
    return f"d{d}_{v}"

def main():
    pl.seed_everything(SEED)

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

    #torch.serialization.add_safe_globals([MNISTModel, MNISTEncoder, AECat, nn.Sequential, nn.Linear, nn.ReLU, nn.LeakyReLU])

    checkpoint_path = './results/mnist_base/CMR/best.ckpt'

    model = MNISTModel.load_from_checkpoint(
        checkpoint_path,
        encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
        rule_module=AECat,
        weights_only=False
    )

    model.eval()
    model.to("cpu")

    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)

    print(f"\nTrain Acc: {train_acc:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    
    rules = model.get_all_rule_vars()
    
    orig_shape = rules.shape

    rules_flat = rules.view(-1, model.n_concepts * 3)

    latent = model.rule_module.rule_encoder(rules_flat)
    reconstructed_flat = model.rule_module.rule_decoder(latent)

    reconstructed_rules = reconstructed_flat.view(orig_shape)

    d_flat = reconstructed_rules.view(-1, 3)  # tasks*rules*concepts, 3
    max_indices_flat = torch.argmax(d_flat, dim=-1)
    temp = torch.zeros_like(d_flat)
    temp[torch.arange(d_flat.size(0)), max_indices_flat] = 1
    reconstructed_rules = temp.view(reconstructed_rules.shape)

    n_tasks, n_rules, n_concepts, _ = rules.shape 
    
    count = 0


    for t in range(n_tasks):
        if VERBOSE:
            print(f"\n" + "="*50)
            print(f" TASK {t}")
            print("="*50)
        
    

        for r in range(n_rules):
            # Extract the specific rule [Concepts, 3]
            orig_rule = rules[t, r]
            recon_rule = reconstructed_rules[t, r]
            
            # Check if they are identical to highlight errors
            is_correct = torch.equal(orig_rule, recon_rule)
            
            info = "" if VERBOSE else f" TASK {t} - "
            status = "V MATCH" if is_correct else "X MISMATCH"
            count += 0 if is_correct else 1
            
            if VERBOSE or not is_correct:
                print(info + f"\n  [Rule {r}] - {status}")
            if not is_correct:
                print(f"  ORIGINAL:\n  {orig_rule}")
                print(f"  RECONSTRUCTED:\n  {recon_rule}")

    print(f"\n  Reconstruction Mismatch Count: {count} \n")


    # task_to_rules, _ = model.aggregate_rules(test_loader, type='most_likely')
    # for task in sorted(task_to_rules.keys()):
    #     print(f"Task {task}: {task_to_rules[task]}")

    model.make_editable()

    
    new_rule = torch.zeros(20, 3)
    new_rule[:, 0] = 1

    model.change_rule(84, new_rule)

    new_rules = model.get_all_rule_vars()


    for t in range(n_tasks):
        if VERBOSE:
            print(f"\n" + "="*50)
            print(f" TASK {t}")
            print("="*50)
        
    

        for r in range(n_rules):
            # Extract the specific rule [Concepts, 3]
            orig_rule = rules[t, r]
            new_rule = new_rules[t, r]
            
            # Check if they are identical to highlight errors
            is_correct = torch.equal(orig_rule, new_rule)
            
            info = "" if VERBOSE else f" TASK {t}"
            status = "V MATCH" if is_correct else "X MISMATCH"
            count += 0 if is_correct else 1
            
            if VERBOSE or not is_correct:
                print(info + f"\n  [Rule {r}] - {status}")
            if not is_correct:
                print(f"  ORIGINAL:\n  {orig_rule}")
                print(f"  EDITED:\n  {new_rule}")

    print(f"\n  Pre-Post Edit Mismatch Count: {count}")

    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)

    print(f"\nOne-change Train Acc: {train_acc:.4f}")
    print(f"One-change Test Acc:  {test_acc:.4f}")

    #Check that eval also changes when adding dumb rules
    new_rule = torch.zeros(20, 3)
    new_rule[:, 0] = 1

    model.change_rule(84, new_rule)

    new_rules = model.get_all_rule_vars()
    
    for i in range(N_RULES * n_tasks):
        model.change_rule(i, new_rule)

    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)

    print(f"\nBS Train Acc: {train_acc:.4f}")
    print(f"BS Test Acc:  {test_acc:.4f}")



if __name__ == "__main__":
    main()