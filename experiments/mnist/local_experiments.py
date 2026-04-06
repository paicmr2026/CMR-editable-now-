import unittest
import gc
from collections import defaultdict
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# import joblib

# from experiments.celeba.models import DNN, CBMLinear, CBMDeep, CEMDeep, StandardDCR
# from experiments.global_mnist_beta.dataset import load_mnist_data
# from experiments.mnist.models_comps import DNN_CNN, CBMLinear_CNN, CBMDeep_CNN, CEMDeep_CNN, StandardDCR_CNN
# from experiments.mnist.models_copy import MNISTModel as MNISTModel_old, InputTypes, get_accuracy, get_concept_accuracy, SaveBestModelCallbackVal
# from experiments.mnist.rule_logger import RuleLogger
from mnist_dataset import create_single_digit_addition, get_mnist_add_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
from model import MNISTModel as MNISTModel_old, InputTypes, get_accuracy, get_concept_accuracy, SaveBestModelCallbackVal, MNISTEncoder, ProbRDCat
from model_simmilarity import (
    MNISTModel, MNISTEncoder, ProbRDCat,
    InputTypes, get_accuracy
)
# from lightning.pytorch.loggers import TensorBoardLogger

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




def c_idx_to_c_sym(c_idx, num_digits, digit_limit, missing_digits=None, digit_prefix=False):
    if missing_digits is None:
        missing_digits = []
    c_names = [*[str(i) for i in range(digit_limit) if i not in missing_digits]] * num_digits
    if digit_prefix:
        return 'd' + str(c_idx // 10) + '_' + c_names[c_idx]
    return c_names[c_idx]

def filter_concepts(c_train, REMOVE_DIGIT_CONCEPTS):
    new_c_train = []
    for example_idx in range(len(c_train)):
        new = []
        for c_idx in range(c_train[example_idx].shape[1]):
            if c_idx not in REMOVE_DIGIT_CONCEPTS:
                new.append(c_train[example_idx][:, c_idx].unsqueeze(-1))
        new_c_train.append(torch.cat(new, dim=-1))
    return new_c_train

def c_idx_to_name(c_idx, num_digits, digit_limit):
    d = c_idx // digit_limit
    v = c_idx % digit_limit
    return f"d{d}_{v}"


class TestCMR(unittest.TestCase):

    def train_old_cmr_mnist(self):
        # n_cv = 3

        n_cv = 1
        results = []

        columns = [
            'seed', 
            'task_train_acc', 
            'task_test_acc', 
            'concepts_train_acc',
            'concepts_test_acc', 
            'rules', 
            'path']
        
        results_dir = f"experiments/mnist/results/mnist_base/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)


        # DATASET
        concept_names, class_names, explanations = create_single_digit_addition(2, 10)
        true_explanations = defaultdict(set)
        nb_ground_truth_rules = 0
        for key in explanations:
            for conj in explanations[key]['explanation'].split(' | '):
                conj = conj.replace('(', '').replace(')', '')  # remove all ( and )
                true_explanations[key].add(conj)
                nb_ground_truth_rules += 1

        x_train, c_train, y_train = get_mnist_add_dataset(True, 2, 10)
        x_test, c_test, y_test = get_mnist_add_dataset(False, 2, 10)

        x_train = torch.stack(x_train, dim=1)
        c_train = torch.cat(c_train, dim=-1)
        y_train = F.one_hot(y_train.unsqueeze(-1).long().ravel()).float()

        x_test = torch.stack(x_test, dim=1)
        c_test = torch.cat(c_test, dim=-1)
        y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()

        val_split = 0.1
        train_set_size = int(len(x_train) * (1 - val_split))
        x_val, c_val, y_val = x_train[train_set_size:], c_train[train_set_size:], y_train[train_set_size:]
        x_train, c_train, y_train = x_train[:train_set_size], c_train[:train_set_size], y_train[:train_set_size]

        train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=128, shuffle=False)
        val_loader = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=128)
        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=128)

        n_concepts = c_train.shape[1]
        n_tasks = y_train.shape[1]   

        for seed in range(n_cv):
            pl.seed_everything(1)

            c2s = lambda c_idx: c_idx_to_c_sym(c_idx, 2, 10, [])

            model = MNISTModel_old(
            encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
            rule_module=ProbRDCat,
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            n_rules=N_RULES,
            rule_emb_size=RULE_EMB,
            emb_size=EMB_SIZE,
            selector_input=InputTypes.concepts,   # or embedding depending on your current implementation
            concept_names=[c2s(i) for i in range(n_concepts)],
            w_c=1,
            w_y=1,
            w_yF=1,
            reset_selector=True,
            reset_selector_every_n_epochs=40,
            lr=LR)

            checkpoint_cb = ModelCheckpoint(dirpath="experiments/mnist/results/mnist_base/CMR/" + str(seed) + "/", save_top_k=1,
                                                monitor="val_loss", mode='min')
            cb = SaveBestModelCallbackVal()
            # max_epochs=5 just for testing, originally 300
            trainer = pl.Trainer(max_epochs=5, callbacks=[cb, checkpoint_cb], deterministic=True,)
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            model.load_state_dict(cb.best_state_dict)
            model.eval()

            train_acc = get_accuracy(model, train_loader)
            test_acc = get_accuracy(model, test_loader)
            train_c_acc = get_concept_accuracy(model, train_loader)
            test_c_acc = get_concept_accuracy(model, test_loader)
            print("Train task accuracy :", train_acc)
            print("Test task accuracy :", test_acc)
            print("Train concept accuracy :", train_c_acc)
            print("Test concept accuracy :", test_c_acc)

            batch = next(iter(test_loader))
            x_batch, c_batch, y_batch = batch
            with torch.no_grad():
                out = model((x_batch, c_batch, y_batch))


            task_to_rules, task_to_rules_idx = model.aggregate_rules(train_loader, type='most_likely')
            for task in task_to_rules:
                print("=== Task", task, "===")
                for rule in task_to_rules[task]:
                    print(rule)

            results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, task_to_rules, checkpoint_cb.best_model_path])
            pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))
        
        del model, trainer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()


    def train_extended_cmr_mnist(self):
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
            dirpath="./results/mnist_base/CMR_rule_selector/",
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

        del model, trainer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()



    def test_rule_selector_rule_switch(self):
        
        pl.seed_everything(1)

        c2s = lambda c_idx: c_idx_to_c_sym(c_idx, 2, 10, [])

        x_test, c_test, y_test = get_mnist_add_dataset(False, 2, 10)

        x_test = torch.stack(x_test, dim=1)
        c_test = torch.cat(c_test, dim=-1)
        y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()

        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=BATCH_SIZE)

        n_concepts = c_test.shape[1]
        n_tasks = y_test.shape[1]

        model = MNISTModel(
            encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
            emb_size=EMB_SIZE,
            rule_emb_size=RULE_EMB,
            n_tasks=n_tasks,
            n_rules=N_RULES,
            n_concepts=n_concepts,
            concept_names=[c_idx_to_name(i, NUM_DIGITS, DIGIT_LIMIT) for i in range(n_concepts)],
            rule_module=ProbRDCat,
            selector_input=InputTypes.embedding
        )

        checkpoint_path = "./results/mnist_base/CMR_rule_selector/best.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Laden info: {msg}")

        model.eval()

        print("\n--- Start Test ---")
        baseline_acc = get_accuracy(model, test_loader)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        results = []

        with torch.no_grad():

            for t_idx in range(n_tasks):
                start = t_idx * model.n_rules
                end = (t_idx + 1) * model.n_rules
                
                # Pak de rule embeddings
                original_weights = model.rule_module.rules.weight.data[start:end].clone()
                # Shuffle de volgorde van de regels binnen deze taak
                perm = torch.randperm(model.n_rules)
                model.rule_module.rules.weight.data[start:end] = original_weights[perm]

                permuted_acc = get_accuracy(model, test_loader)
                print(f"Accuracy na verplaatsen regels: {permuted_acc:.4f}")
                
                diff = baseline_acc - permuted_acc
                print(f"Verschil: {diff:.4f}")

                model.rule_module.rules.weight.data[start:end] = original_weights

                results.append({
                    "task": t_idx,
                    "baseline_acc": baseline_acc,
                    "permuted_acc": permuted_acc,
                    "accuracy_diff": diff
                })

        results_dir = "./results/rule_switch_tests/rule_selector"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "rule_selector_results.csv")
        pd.DataFrame(results).to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")

        del model, trainer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
            



    def test_extended_cmr_rule_switch(self):
        pl.seed_everything(1)

        x_test, c_test, y_test = get_mnist_add_dataset(False, 2, 10)

        x_test = torch.stack(x_test, dim=1)
        c_test = torch.cat(c_test, dim=-1)
        y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()

        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=BATCH_SIZE)

        n_concepts = c_test.shape[1]
        n_tasks = y_test.shape[1]

        # ---------------- OLD CMR ----------------
        old_model = MNISTModel_old(
            encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
            rule_module=ProbRDCat,
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            n_rules=N_RULES,
            rule_emb_size=RULE_EMB,
            emb_size=EMB_SIZE,
            selector_input=InputTypes.concepts,
            concept_names=[c_idx_to_name(i, NUM_DIGITS, DIGIT_LIMIT) for i in range(n_concepts)],
            w_c=1,
            w_y=1,
            w_yF=1,
            reset_selector=True,
            reset_selector_every_n_epochs=40,
            lr=LR
        )

        old_checkpoint_path = "./results/mnist_base/CMR/best.ckpt"
        old_checkpoint = torch.load(old_checkpoint_path, map_location="cpu", weights_only=False)
        old_msg = old_model.load_state_dict(old_checkpoint['state_dict'], strict=False)
        print(f"Old CMR loaded info: {old_msg}")

        old_model.eval()

        old_baseline_acc = get_accuracy(old_model, test_loader)
        print(f"\nOld CMR Baseline Accuracy: {old_baseline_acc:.4f}")

        # ---------------- EXTENDED CMR ----------------
        extended_model = MNISTModel(
            encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
            emb_size=EMB_SIZE,
            rule_emb_size=RULE_EMB,
            n_tasks=n_tasks,
            n_rules=N_RULES,
            n_concepts=n_concepts,
            concept_names=[c_idx_to_name(i, NUM_DIGITS, DIGIT_LIMIT) for i in range(n_concepts)],
            rule_module=ProbRDCat,
            selector_input=InputTypes.embedding,
            lr=LR,
            reset_selector=True,
            reset_selector_every_n_epochs=40,
            w_c=1,
            w_y=1,
            w_yF=1,
        )

        extended_checkpoint_path = "./results/mnist_base/CMR_rule_selector/best.ckpt"
        extended_checkpoint = torch.load(extended_checkpoint_path, map_location="cpu", weights_only=False)
        extended_msg = extended_model.load_state_dict(extended_checkpoint['state_dict'], strict=False)
        print(f"Extended CMR loaded info: {extended_msg}")

        extended_model.eval()

        extended_baseline_acc = get_accuracy(extended_model, test_loader)
        print(f"Extended CMR Baseline Accuracy: {extended_baseline_acc:.4f}")

        results = []

        with torch.no_grad():
            for t_idx in range(n_tasks):
                start = t_idx * N_RULES
                end = (t_idx + 1) * N_RULES

                # Save original weights
                old_original_weights = old_model.rule_module.rules.weight.data[start:end].clone()
                extended_original_weights = extended_model.rule_module.rules.weight.data[start:end].clone()

                # Same permutation for both models
                perm = torch.randperm(N_RULES)

                # Apply human-like edit
                old_model.rule_module.rules.weight.data[start:end] = old_original_weights[perm]
                extended_model.rule_module.rules.weight.data[start:end] = extended_original_weights[perm]

                # Evaluate
                old_permuted_acc = get_accuracy(old_model, test_loader)
                extended_permuted_acc = get_accuracy(extended_model, test_loader)

                old_diff = old_baseline_acc - old_permuted_acc
                extended_diff = extended_baseline_acc - extended_permuted_acc
                improvement = extended_permuted_acc - old_permuted_acc

                print(f"\nTask {t_idx}")
                print(f"Old CMR after edit:       {old_permuted_acc:.4f} (diff: {old_diff:.4f})")
                print(f"Extended CMR after edit:  {extended_permuted_acc:.4f} (diff: {extended_diff:.4f})")
                print(f"Extended improvement:     {improvement:.4f}")

                results.append({
                    "task": t_idx,
                    "old_baseline_acc": old_baseline_acc,
                    "old_permuted_acc": old_permuted_acc,
                    "old_accuracy_diff": old_diff,
                    "extended_baseline_acc": extended_baseline_acc,
                    "extended_permuted_acc": extended_permuted_acc,
                    "extended_accuracy_diff": extended_diff,
                    "extended_improvement_over_old": improvement
                })

                # Restore original rule weights
                old_model.rule_module.rules.weight.data[start:end] = old_original_weights
                extended_model.rule_module.rules.weight.data[start:end] = extended_original_weights

        results_dir = "./results/rule_switch_tests/extended_cmr"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "extended_cmr_vs_old_results.csv")
        pd.DataFrame(results).to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")

        del model, trainer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
