import sys
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl

# 1. De "ModuleNotFoundError" fix (mocht je checkpoint oude paden bevatten)
import model_simmilarity
sys.modules['experiments.mnist.model_simmilarity_only'] = model_simmilarity

from model_simmilarity import (
    MNISTModel, MNISTEncoder, ProbRDCat,
    InputTypes, get_accuracy
)
from experiments.mnist.mnist_dataset import addition_dataset

# Gebruik dezelfde config als in je training script
NUM_DIGITS   = 2
DIGIT_LIMIT  = 10
EMB_SIZE     = 500
RULE_EMB     = 1000
N_RULES      = 20
BATCH_SIZE   = 512
SEED         = 42

def c_idx_to_name(c_idx, num_digits, digit_limit):
    d = c_idx // digit_limit
    v = c_idx % digit_limit
    return f"d{d}_{v}"

def main():
    pl.seed_everything(SEED)

    # ── Dataset Laden (Alleen Test) ──────────────────────────────────────────
    _, _, _ = addition_dataset(True, NUM_DIGITS, DIGIT_LIMIT) # Train niet nodig, maar vaak nodig voor initialisatie
    x_test_raw, c_test_raw, y_test_raw = addition_dataset(False, NUM_DIGITS, DIGIT_LIMIT)

    x_test = torch.stack(x_test_raw, dim=1).float()
    c_test = torch.cat(c_test_raw, dim=-1).float()
    y_test = F.one_hot(y_test_raw.unsqueeze(-1).long().ravel()).float()
    test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=BATCH_SIZE)

    n_concepts = c_test.shape[1]
    n_tasks    = y_test.shape[1]

    # ── Model Initialiseren ──────────────────────────────────────────────────
    # We maken een model met de HUIDIGE code architectuur
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

    # ── Gewichten Laden ──────────────────────────────────────────────────────
    checkpoint_path = "./results/mnist_base/CMR/best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Gebruik strict=False om mismatches in lagen (zoals input_emb_proj) te negeren
    # Het laadt wat het kan vinden, de rest blijft zoals geïnitialiseerd
    msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"Laden info: {msg}")

    model.eval()

    # ── De Test: Baseline vs Permutatie ──────────────────────────────────────
    print("\n--- Start Test ---")
    baseline_acc = get_accuracy(model, test_loader)
    print(f"Baseline Accuracy: {baseline_acc:.4f}")

    # Permuteer de regels per taak
    with torch.no_grad():
        for t_idx in range(n_tasks):
            start = t_idx * model.n_rules
            end = (t_idx + 1) * model.n_rules
            
            # Pak de rule embeddings
            weights = model.rule_module.rules.weight.data[start:end]
            # Shuffle de volgorde van de regels binnen deze taak
            perm = torch.randperm(model.n_rules)
            model.rule_module.rules.weight.data[start:end] = weights[perm]

    permuted_acc = get_accuracy(model, test_loader)
    print(f"Accuracy na verplaatsen regels: {permuted_acc:.4f}")
    
    diff = baseline_acc - permuted_acc
    print(f"Verschil: {diff:.4f}")

if __name__ == "__main__":
    main()