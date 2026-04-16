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
from mnist_dataset import create_single_digit_addition, addition_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
from model import MNISTModel as MNISTModel_old, InputTypes, get_accuracy, get_concept_accuracy, SaveBestModelCallbackVal, MNISTEncoder, ProbRDCat
from model_similarity import (
    MNISTModel, MNISTEncoder, ProbRDCat,
    InputTypes, get_accuracy, SimilarityTypes
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


def get_mnist_addition_loaders(batch_size, val_split=0.1, shuffle_train=True):
    x_train_raw, c_train_raw, y_train_raw = addition_dataset(True, NUM_DIGITS, DIGIT_LIMIT)
    x_test_raw,  c_test_raw,  y_test_raw  = addition_dataset(False, NUM_DIGITS, DIGIT_LIMIT)

    x_train = torch.stack(x_train_raw, dim=1).float()
    c_train = torch.cat(c_train_raw, dim=-1).float()
    y_train = F.one_hot(y_train_raw.unsqueeze(-1).long().ravel()).float()

    x_test = torch.stack(x_test_raw, dim=1).float()
    c_test = torch.cat(c_test_raw, dim=-1).float()
    y_test = F.one_hot(y_test_raw.unsqueeze(-1).long().ravel()).float()

    split = int(len(x_train) * (1 - val_split))
    x_val,  c_val,  y_val  = x_train[split:], c_train[split:], y_train[split:]
    x_train, c_train, y_train = x_train[:split], c_train[:split], y_train[:split]

    train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(TensorDataset(x_val,   c_val,   y_val),   batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(x_test,  c_test,  y_test),  batch_size=batch_size)

    return train_loader, val_loader, test_loader, c_train.shape[1], y_train.shape[1]


def create_model(n_concepts, n_tasks, similarity=None, selector_input=InputTypes.embedding):
    model = MNISTModel(
        encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
        emb_size=EMB_SIZE,
        rule_emb_size=RULE_EMB,
        n_tasks=n_tasks,
        n_rules=N_RULES,
        n_concepts=n_concepts,
        concept_names=[c_idx_to_name(i, NUM_DIGITS, DIGIT_LIMIT) for i in range(n_concepts)],
        rule_module=ProbRDCat,
        selector_input=selector_input,
        selector_similarity=similarity if similarity else SimilarityTypes.cosine,
        lr=LR,
        reset_selector=True,
        reset_selector_every_n_epochs=40,
        w_c=1, w_y=1, w_yF=1,
    )

    return model

def train_model(model, train_loader, val_loader, max_epochs=MAX_EPOCHS):
    cb = SaveBestModelCallbackVal()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[cb],
        deterministic=True,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm"
    )

    trainer.fit(model, train_loader, val_loader)

    model.load_state_dict(cb.best_state_dict)
    model.eval()
    model.to("cpu")

    return model


def load_trained_model(checkpoint_path, n_concepts, n_tasks):
    model = create_model(n_concepts, n_tasks)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()
    return model


def evaluate_model(model, train_loader, test_loader):
    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)
    return train_acc, test_acc



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

        x_train, c_train, y_train = addition_dataset(True, 2, 10)
        x_test, c_test, y_test = addition_dataset(False, 2, 10)

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

        train_loader, val_loader, test_loader, n_concepts, n_tasks = get_mnist_addition_loaders(BATCH_SIZE, VAL_SPLIT)

        model = create_model(n_concepts, n_tasks)

        model = train_model(model, train_loader, val_loader, MAX_EPOCHS)

        train_acc, test_acc = evaluate_model(model, train_loader, test_loader)

        print(f"Train Acc: {train_acc:.4f}")
        print(f"Test Acc:  {test_acc:.4f}")

        task_to_rules, _ = model.aggregate_rules(train_loader, type='most_likely')
        for task in sorted(task_to_rules.keys()):
            print(f"Task {task}: {task_to_rules[task]}")

        del model
        gc.collect()
        torch.cuda.empty_cache()


    def test_similarity_metrics_rule_selector(self):
        pl.seed_everything(SEED)

        similarity_metrics = {
            "cosine": SimilarityTypes.cosine,
            "dotproduct": SimilarityTypes.dotproduct,
            "euclidean": SimilarityTypes.euclidean,
        }

        results = []

        train_loader, val_loader, test_loader, n_concepts, n_tasks = get_mnist_addition_loaders(BATCH_SIZE, VAL_SPLIT)

        for sim_name, sim_type in similarity_metrics.items():
            print(f"=== Testing similarity: {sim_name} ===")

            """
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
                selector_similarity=sim_type,
                reset_selector=True,
                reset_selector_every_n_epochs=40,
                w_c=1, w_y=1, w_yF=1,
            )

            cb = SaveBestModelCallbackVal()

            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                callbacks=[cb],
                deterministic=True,
                gradient_clip_val=0.5,
                gradient_clip_algorithm="norm"
            )

            trainer.fit(model, train_loader, val_loader)

            model.load_state_dict(cb.best_state_dict)
            model.eval()
            model.to("cpu")

            train_acc = get_accuracy(model, train_loader)
            test_acc  = get_accuracy(model, test_loader)
            """

            model = create_model(n_concepts, n_tasks, similarity=sim_type)

            model = train_model(model, train_loader, val_loader)

            train_acc, test_acc = evaluate_model(model, train_loader, test_loader)

            print(f"{sim_name} -> Train: {train_acc:.4f}, Test: {test_acc:.4f}")

            results.append({
                "similarity": sim_name,
                "train_acc": train_acc,
                "test_acc": test_acc
            })

            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()

        df = pd.DataFrame(results)
        print("=== FINAL RESULTS ===")
        print(df.sort_values(by="test_acc", ascending=False))

        os.makedirs("results/similarity_tests/", exist_ok=True)
        df.to_csv("results/similarity_tests/similarity_results.csv", index=False)


    def test_rule_modification(self):
        pl.seed_everything(SEED)

        print("=== TEST: Rule Modification ===")

        train_loader, _, test_loader, n_concepts, n_tasks = get_mnist_addition_loaders(BATCH_SIZE, VAL_SPLIT)

        model = load_trained_model(
            "./results/mnist_base/CMR_rule_selector/best.ckpt",
            n_concepts,
            n_tasks
        )

        baseline_acc = get_accuracy(model, test_loader)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        results = []

        with torch.no_grad():
            for t_idx in range(n_tasks):
                start = t_idx * N_RULES
                end   = (t_idx + 1) * N_RULES

                original_rules = model.rule_module.rules.weight.data[start:end].clone()

                rule_idx = torch.randint(0, N_RULES, (1,)).item()

                modification = 0.1 * torch.randn_like(original_rules[rule_idx])
                model.rule_module.rules.weight.data[start + rule_idx] += modification

                modified_acc = get_accuracy(model, test_loader)

                diff = baseline_acc - modified_acc

                print(f"Task {t_idx}, Modified Rule {rule_idx}")
                print(f"Modified Accuracy: {modified_acc:.4f}")
                print(f"Accuracy Drop: {diff:.4f}")

                """
                # hoeveel predictions zijn veranderd? (optioneel?)
                changed_predictions = 0
                total = 0

                for batch in test_loader:
                    x_batch, c_batch, y_batch = batch
                    out_before = model((x_batch, c_batch, y_batch))

                    model.rule_module.rules.weight.data[start:end] = original_rules
                    out_baseline = model((x_batch, c_batch, y_batch))

                    model.rule_module.rules.weight.data[start + rule_idx] += modification

                    pred_before = out_before.argmax(dim=1)
                    pred_after  = out_baseline.argmax(dim=1)

                    changed_predictions += (pred_before != pred_after).sum().item()
                    total += pred_before.size(0)

                change_ratio = changed_predictions / total
                print(f"Prediction Change Ratio: {change_ratio:.4f}")

                results.append({
                    "task": t_idx,
                    "rule_idx": rule_idx,
                    "baseline_acc": baseline_acc,
                    "modified_acc": modified_acc,
                    "accuracy_drop": diff,
                    "prediction_change_ratio": change_ratio
                })
                """
                model.rule_module.rules.weight.data[start:end] = original_rules

                results.append({
                    "task": t_idx,
                    "rule_idx": rule_idx,
                    "baseline_acc": baseline_acc,
                    "modified_acc": modified_acc,
                    "accuracy_drop": diff,
                })

        os.makedirs("results/rule_modification/", exist_ok=True)
        pd.DataFrame(results).to_csv("results/rule_modification/results.csv", index=False)

        print("Results saved to results/rule_modification/results.csv")

        del model, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()


    def test_rule_deletion(self):
        pl.seed_everything(SEED)

        print("=== TEST: Rule Deletion ===")

        train_loader, _, test_loader, n_concepts, n_tasks = get_mnist_addition_loaders(BATCH_SIZE, VAL_SPLIT)

        model = MNISTModel(
            encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
            emb_size=EMB_SIZE,
            rule_emb_size=RULE_EMB,
            n_tasks=n_tasks,
            n_rules=N_RULES,
            n_concepts=n_concepts,
            concept_names=[c_idx_to_name(i, NUM_DIGITS, DIGIT_LIMIT) for i in range(n_concepts)],
            rule_module=ProbRDCat,
            selector_input=InputTypes.embedding,
            selector_similarity=SimilarityTypes.cosine,
        )

        model = load_trained_model(
            "./results/mnist_base/CMR_rule_selector/best.ckpt",
            n_concepts,
            n_tasks
        )

        baseline_acc = get_accuracy(model, test_loader)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        results = []

        with torch.no_grad():
            for t_idx in range(n_tasks):
                start = t_idx * N_RULES
                end   = (t_idx + 1) * N_RULES

                original_rules = model.rule_module.rules.weight.data[start:end].clone()

                rule_idx = torch.randint(0, N_RULES, (1,)).item()

                print(f"Task {t_idx}, Deleting Rule {rule_idx}")

                model.rule_module.rules.weight.data[start + rule_idx] = 0.0

                deleted_acc = get_accuracy(model, test_loader)
                acc_drop = baseline_acc - deleted_acc

                print(f"Accuracy after deletion: {deleted_acc:.4f}")
                print(f"Accuracy drop: {acc_drop:.4f}")

                """
                changed_predictions = 0
                total = 0

                new_preds = []

                for batch in test_loader:
                    x_batch, c_batch, y_batch = batch
                    out = model((x_batch, c_batch, y_batch))
                    new_preds.append(out.argmax(dim=1))

                new_preds = torch.cat(new_preds)

                changed = (baseline_preds != new_preds)
                changed_predictions = changed.sum().item()
                total = len(baseline_preds)

                change_ratio = changed_predictions / total

                print(f"Prediction Change Ratio: {change_ratio:.4f}")

                results.append({
                    "task": t_idx,
                    "rule_idx": rule_idx,
                    "baseline_acc": baseline_acc,
                    "deleted_acc": deleted_acc,
                    "accuracy_drop": acc_drop,
                    "prediction_change_ratio": change_ratio
                })
                """

                results.append({
                    "task": t_idx,
                    "rule_idx": rule_idx,
                    "baseline_acc": baseline_acc,
                    "deleted_acc": deleted_acc,
                    "accuracy_drop": acc_drop,
                })

                model.rule_module.rules.weight.data[start:end] = original_rules

        os.makedirs("results/rule_deletion/", exist_ok=True)
        pd.DataFrame(results).to_csv("results/rule_deletion/results.csv", index=False)

        print("Results saved to results/rule_deletion/results.csv")

        del model, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()


    def test_rule_addition(self):
        pl.seed_everything(SEED)

        print("=== TEST: Rule Addition ===")

        train_loader, _, test_loader, n_concepts, n_tasks = get_mnist_addition_loaders(BATCH_SIZE, VAL_SPLIT)

        model = load_trained_model(
            "./results/mnist_base/CMR_rule_selector/best.ckpt",
            n_concepts,
            n_tasks
        )

        baseline_acc = get_accuracy(model, test_loader)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        results = []

        with torch.no_grad():
            for t_idx in range(n_tasks):
                start = t_idx * N_RULES
                end   = (t_idx + 1) * N_RULES

                original_rules = model.rule_module.rules.weight.data[start:end].clone()

                base_rule_idx = torch.randint(0, N_RULES, (1,)).item()

                new_rule = original_rules[base_rule_idx].clone()

                new_rule += 0.05 * torch.randn_like(new_rule)

                replace_idx = torch.randint(0, N_RULES, (1,)).item()
                model.rule_module.rules.weight.data[start + replace_idx] = new_rule

                print(f"Task {t_idx}")
                print(f"Base rule: {base_rule_idx} → inserted at {replace_idx}")

                new_acc = get_accuracy(model, test_loader)
                acc_diff = new_acc - baseline_acc

                print(f"New Accuracy: {new_acc:.4f}")
                print(f"Accuracy Change: {acc_diff:.4f}")

                selected_count = 0
                total = 0

                for batch in test_loader:
                    x_batch, c_batch, y_batch = batch

                    outputs = model((x_batch, c_batch, y_batch))

                    if hasattr(model, "selector_scores"):
                        scores = model.selector_scores

                        selected = scores.argmax(dim=1)
                        selected_count += (selected == replace_idx).sum().item()
                        total += selected.size(0)

                usage_ratio = selected_count / total if total > 0 else 0

                print(f"New Rule Usage Ratio: {usage_ratio:.4f}")

                results.append({
                    "task": t_idx,
                    "base_rule": base_rule_idx,
                    "inserted_rule_idx": replace_idx,
                    "baseline_acc": baseline_acc,
                    "new_acc": new_acc,
                    "accuracy_change": acc_diff,
                    "usage_ratio": usage_ratio
                })

                model.rule_module.rules.weight.data[start:end] = original_rules
        os.makedirs("results/rule_addition/", exist_ok=True)
        pd.DataFrame(results).to_csv("results/rule_addition/results.csv", index=False)

        print("Results saved to results/rule_addition/results.csv")

        del model, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()


    def test_out_of_distribution_robustness(self):
        pl.seed_everything(SEED)

        print("=== TEST: Out-of-Distribution Robustness ===")

        _, _, test_loader, n_concepts, n_tasks = get_mnist_addition_loaders(BATCH_SIZE, VAL_SPLIT)

        model = load_trained_model(
            "./results/mnist_base/CMR_rule_selector/best.ckpt",
            n_concepts,
            n_tasks
        )

        baseline_embeddings = []

        with torch.no_grad():
            for batch in test_loader:
                x_batch, c_batch, y_batch = batch
                emb = model.encoder(x_batch)
                baseline_embeddings.append(emb)

        baseline_embeddings = torch.cat(baseline_embeddings)

        results = []

        with torch.no_grad():
            for t_idx in range(n_tasks):
                start = t_idx * N_RULES
                end   = (t_idx + 1) * N_RULES

                original_rules = model.rule_module.rules.weight.data[start:end].clone()

                new_rule = torch.randn_like(original_rules[0])  # random = unseen distribution

                replace_idx = torch.randint(0, N_RULES, (1,)).item()
                model.rule_module.rules.weight.data[start + replace_idx] = new_rule

                print(f"Task {t_idx}: Injected OOD rule at index {replace_idx}")

                new_embeddings = []

                for batch in test_loader:
                    x_batch, c_batch, y_batch = batch
                    emb = model.encoder(x_batch)
                    new_embeddings.append(emb)

                new_embeddings = torch.cat(new_embeddings)

                cosine_sim = F.cosine_similarity(baseline_embeddings, new_embeddings, dim=1).mean().item()

                l2_dist = torch.norm(baseline_embeddings - new_embeddings, dim=1).mean().item()

                print(f"Cosine similarity: {cosine_sim:.4f}")
                print(f"L2 distance:       {l2_dist:.4f}")

                results.append({
                    "task": t_idx,
                    "replaced_rule_idx": replace_idx,
                    "cosine_similarity": cosine_sim,
                    "l2_distance": l2_dist,
                })

                model.rule_module.rules.weight.data[start:end] = original_rules

        os.makedirs("results/ood_robustness/", exist_ok=True)
        pd.DataFrame(results).to_csv("results/ood_robustness/results.csv", index=False)

        print("Results saved to results/ood_robustness/results.csv")

        del model, test_loader
        gc.collect()
        torch.cuda.empty_cache()


    """
    def test_rule_selector_rule_switch(self):
        
        pl.seed_everything(1)

        c2s = lambda c_idx: c_idx_to_c_sym(c_idx, 2, 10, [])

        x_test, c_test, y_test = addition_dataset(False, 2, 10)

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

        print("--- Start Test ---")
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
        print(f"Results saved to {results_file}")

        del model, trainer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
            



    def test_extended_cmr_rule_switch(self):
        pl.seed_everything(1)

        x_test, c_test, y_test = addition_dataset(False, 2, 10)

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
        print(f"Old CMR Baseline Accuracy: {old_baseline_acc:.4f}")

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

                print(f"Task {t_idx}")
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
        print(f"Results saved to {results_file}")

        del model, trainer, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
    """
