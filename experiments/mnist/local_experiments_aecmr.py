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
import sys
import pandas as pd
#from models_copy import MNISTModel as MNISTModel_old, InputTypes, get_accuracy, get_concept_accuracy, SaveBestModelCallbackVal, MNISTEncoder, ProbRDCat
from model_similarity import (
    MNISTModel as MNISTModel_sim, MNISTEncoder as MNISTEncoder_sim, ProbRDCat as ProbRDCat_sim,
    InputTypes, get_accuracy, SimilarityTypes
)
from autoencoderCMR import (
    MNISTModel, MNISTEncoder, AECat,
    InputTypes, SimilarityTypes, SaveBestModelCallbackVal,
    get_accuracy, get_concept_accuracy
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



class AECMRTest(unittest.TestCase):

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

            model = MNISTModel_sim(
                encoder=MNISTEncoder_sim(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
                emb_size=EMB_SIZE,
                rule_emb_size=RULE_EMB,
                n_tasks=n_tasks,
                n_rules=N_RULES,
                n_concepts=n_concepts,
                concept_names=[c_idx_to_name(i, NUM_DIGITS, DIGIT_LIMIT) for i in range(n_concepts)],
                rule_module=ProbRDCat_sim,
                lr=LR,
                selector_input=InputTypes.embedding,
                selector_similarity=sim_type, 
                temperature=0.1,
                reset_selector=True,
                reset_selector_every_n_epochs=40,
                w_c=1, w_y=1, w_yF=1,
            )

            cb = SaveBestModelCallbackVal()

            trainer = pl.Trainer(
                max_epochs=50,
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
            final_val_loss = model.val_info["loss"][-1]

            print(f"{sim_name} -> Train: {train_acc:.4f}, Test: {test_acc:.4f}")

            results.append({
                "similarity": sim_name,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "final_loss": final_val_loss
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

        model = MNISTModel(
            encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
            emb_size=EMB_SIZE,
            rule_emb_size=RULE_EMB,
            n_tasks=n_tasks,
            n_rules=N_RULES,
            n_concepts=n_concepts,
            concept_names=[c_idx_to_name(i, NUM_DIGITS, DIGIT_LIMIT) for i in range(n_concepts)],
            rule_module=AECat,
            lr=LR,
            selector_input=InputTypes.embedding,
            selector_similarity=SimilarityTypes.cosine, #Kies tussen cosine, dotproduct, euclidean
            temperature=0.1,
            reset_selector=True,
            reset_selector_every_n_epochs=40,
            w_c=1, w_y=1, w_yF=1,
        )

        checkpoint = torch.load("results/mnist_base/CMR/best.ckpt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        model.eval()

        baseline_acc = get_accuracy(model, test_loader)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        results = []

        scale = 8

        with torch.no_grad():
            for t_idx in range(n_tasks):
                start = t_idx * N_RULES
                end   = (t_idx + 1) * N_RULES

                original_rules = model.rule_module.rules.weight.data[start:end].clone()

                all_p_s = []
                for batch in test_loader:
                    _, _, _, _, p_s, _, _ = model(batch)
                    all_p_s.append(p_s[:, t_idx, :])

                avg_usage = torch.cat(all_p_s, dim=0).mean(dim=0)  # (rule,)

                rule_idx_argmax = avg_usage.argmax().item()

                probs = avg_usage / avg_usage.sum()
                rule_idx_sampled = torch.multinomial(probs, 1).item()

                for strategy_name, rule_idx in [("argmax", rule_idx_argmax), ("sampled", rule_idx_sampled)]:
                    #modification = 0.1 * torch.randn_like(original_rules[rule_idx])
                    modification = scale * torch.randn_like(original_rules[rule_idx])
                    model.rule_module.rules.weight.data[start + rule_idx] += modification
                    #model.rule_module.rules.weight.data[start + rule_idx] = 0

                    modified_acc = get_accuracy(model, test_loader)
                    diff = baseline_acc - modified_acc

                    print(f"Task {t_idx}, Strategy {strategy_name}, Rule {rule_idx}")
                    print(f"Modified Accuracy: {modified_acc:.4f}")
                    print(f"Accuracy Drop: {diff:.4f}")
                
                    # hoeveel predictions zijn veranderd? (optioneel?)
                    changed_predictions = 0
                    total = 0

                    for batch in test_loader:
                        x_batch, c_batch, y_batch = batch
                        #out_before = model((x_batch, c_batch, y_batch))
                        _, _, p_y_mod, _, p_s_mod, _, _ = model((x_batch, c_batch, y_batch))

                        model.rule_module.rules.weight.data[start:end] = original_rules
                        #out_baseline = model((x_batch, c_batch, y_batch))
                        _, _, p_y_base, _, p_s_base, _, _ = model((x_batch, c_batch, y_batch))

                        model.rule_module.rules.weight.data[start + rule_idx] += modification

                        y_mod  = torch.einsum("btr,btr->bt", p_s_mod, p_y_mod)
                        y_base = torch.einsum("btr,btr->bt", p_s_base, p_y_base)

                        pred_mod  = (y_mod > 0.5)
                        pred_base = (y_base > 0.5)

                        changed_predictions += (pred_mod != pred_base).sum().item()
                        total += pred_base.size(0)

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
                    
                    model.rule_module.rules.weight.data[start:end] = original_rules

                    results.append({
                        "task": t_idx,
                        "rule_idx": rule_idx,
                        "baseline_acc": baseline_acc,
                        "modified_acc": modified_acc,
                        "accuracy_drop": diff,
                    })

        os.makedirs("results/rule_modification/", exist_ok=True)
        pd.DataFrame(results).to_csv(f"results/rule_modification/results_scale_{scale}.csv", index=False)

        print(f"Results saved to results/rule_modification/results_scale_{scale}.csv")

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
            rule_module=AECat,
            lr=LR,
            selector_input=InputTypes.embedding,
            selector_similarity=SimilarityTypes.cosine, #Kies tussen cosine, dotproduct, euclidean
            temperature=0.1,
            reset_selector=True,
            reset_selector_every_n_epochs=40,
            w_c=1, w_y=1, w_yF=1,
        )

        checkpoint = torch.load("results/mnist_base/CMR/best.ckpt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        model.eval()

        baseline_acc = get_accuracy(model, test_loader)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        results = []

        with torch.no_grad():
            for t_idx in range(n_tasks):
                start = t_idx * N_RULES
                end   = (t_idx + 1) * N_RULES

                original_rules = model.rule_module.rules.weight.data[start:end].clone()

                all_p_s = []
                for batch in test_loader:
                    _, _, _, _, p_s, _, _ = model(batch)
                    all_p_s.append(p_s[:, t_idx, :])

                avg_usage = torch.cat(all_p_s, dim=0).mean(dim=0)  # (rule,)

                rule_idx_argmax = avg_usage.argmax().item()

                probs = avg_usage / avg_usage.sum()
                rule_idx_sampled = torch.multinomial(probs, 1).item()

                for strategy_name, rule_idx in [("argmax", rule_idx_argmax), ("sampled", rule_idx_sampled)]:
                    model.rule_module.rules.weight.data[start + rule_idx] = 0.0

                    modified_acc = get_accuracy(model, test_loader)
                    diff = baseline_acc - modified_acc

                    print(f"Task {t_idx}, Strategy {strategy_name}, Rule {rule_idx}")
                    print(f"Modified Accuracy: {modified_acc:.4f}")
                    print(f"Accuracy Drop: {diff:.4f}")
                
                    # hoeveel predictions zijn veranderd? (optioneel?)
                    changed_predictions = 0
                    total = 0

                    for batch in test_loader:
                        x_batch, c_batch, y_batch = batch
                        #out_before = model((x_batch, c_batch, y_batch))
                        _, _, p_y_mod, _, p_s_mod, _, _ = model((x_batch, c_batch, y_batch))

                        model.rule_module.rules.weight.data[start:end] = original_rules
                        #out_baseline = model((x_batch, c_batch, y_batch))
                        _, _, p_y_base, _, p_s_base, _, _ = model((x_batch, c_batch, y_batch))

                        model.rule_module.rules.weight.data[start + rule_idx] = 0.0

                        y_mod  = torch.einsum("btr,btr->bt", p_s_mod, p_y_mod)
                        y_base = torch.einsum("btr,btr->bt", p_s_base, p_y_base)

                        pred_mod  = (y_mod > 0.5)
                        pred_base = (y_base > 0.5)

                        changed_predictions += (pred_mod != pred_base).sum().item()
                        total += pred_base.size(0)

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
                    
                    model.rule_module.rules.weight.data[start:end] = original_rules

                    results.append({
                        "task": t_idx,
                        "rule_idx": rule_idx,
                        "baseline_acc": baseline_acc,
                        "modified_acc": modified_acc,
                        "accuracy_drop": diff,
                    })

        os.makedirs("results/rule_deletion/", exist_ok=True)
        pd.DataFrame(results).to_csv(f"results/rule_deletion/results.csv", index=False)

        print(f"Results saved to results/rule_deletion/results.csv")

        del model, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()


    def test_rule_addition(self):
        pl.seed_everything(SEED)

        print("=== TEST: Rule Addition ===")

        train_loader, _, test_loader, n_concepts, n_tasks = get_mnist_addition_loaders(BATCH_SIZE, VAL_SPLIT)

        model = MNISTModel(
            encoder=MNISTEncoder(emb_size=EMB_SIZE, cp_output=DIGIT_LIMIT, number_digits=NUM_DIGITS),
            emb_size=EMB_SIZE,
            rule_emb_size=RULE_EMB,
            n_tasks=n_tasks,
            n_rules=N_RULES,
            n_concepts=n_concepts,
            concept_names=[c_idx_to_name(i, NUM_DIGITS, DIGIT_LIMIT) for i in range(n_concepts)],
            rule_module=AECat,
            lr=LR,
            selector_input=InputTypes.embedding,
            selector_similarity=SimilarityTypes.cosine, #Kies tussen cosine, dotproduct, euclidean
            temperature=0.1,
            reset_selector=True,
            reset_selector_every_n_epochs=40,
            w_c=1, w_y=1, w_yF=1,
        )

        checkpoint = torch.load("results/mnist_base/CMR/best.ckpt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        model.eval()

        baseline_acc = get_accuracy(model, test_loader)
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        results = []

        scale = 12

        with torch.no_grad():
            for t_idx in range(n_tasks):
                start = t_idx * N_RULES
                end   = (t_idx + 1) * N_RULES

                original_rules = model.rule_module.rules.weight.data[start:end].clone()

                base_rule_idx = torch.randint(0, N_RULES, (1,)).item()

                new_rule = original_rules[base_rule_idx].clone()

                new_rule += scale * torch.randn_like(new_rule)

                insert_idx = torch.randint(0, N_RULES, (1,)).item()
                modified_block = torch.cat([
                    original_rules[:insert_idx],
                    new_rule.unsqueeze(0),
                    original_rules[insert_idx:-1]   # drop last to keep size fixed
                ], dim=0)

                model.rule_module.rules.weight.data[start:end] = modified_block

                print(f"Task {t_idx}")
                print(f"Base rule: {base_rule_idx} → inserted at {insert_idx}")

                new_acc = get_accuracy(model, test_loader)
                acc_diff = new_acc - baseline_acc

                print(f"New Accuracy: {new_acc:.4f}")
                print(f"Accuracy Change: {acc_diff:.4f}")

                selected_count = 0
                total = 0

                all_p_s = []

                for batch in test_loader:
                    x_batch, c_batch, y_batch = batch

                    _, _, _, _, log_p_s, _, _ = model((x_batch, c_batch, y_batch))

                    p_s = torch.softmax(log_p_s, dim=-1)

                    all_p_s.append(p_s[:, t_idx, :])

                all_p_s = torch.cat(all_p_s, dim=0)

                usage_ratio = all_p_s[:, insert_idx].mean().item()

                print(f"New Rule Usage Ratio: {usage_ratio:.4f}")

                #entropy = -(all_p_s * torch.log(all_p_s + 1e-8)).sum(dim=-1).mean().item()
                #print(f"Selector Entropy: {entropy:.4f}")

                results.append({
                    "task": t_idx,
                    "base_rule": base_rule_idx,
                    "inserted_rule_idx": insert_idx,
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
