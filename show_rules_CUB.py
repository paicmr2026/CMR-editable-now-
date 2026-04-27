import torch
import os
import lightning.pytorch as pl
from experiments.mnist.autoencoderCMR import MNISTModel, AECat, InputTypes
from experiments.cub.models import CUBEncoder
from experiments.cub.CUB200.cub_loader import load_data, CONCEPT_SEMANTICS, CLASS_NAMES

# Config - must match training
EMB_SIZE = 100
RULE_EMB = 500
N_RULES = 3
BATCH_SIZE = 64
CHECKPOINT_PATH = './results/cub_base/CMR/best_cub_cmr.ckpt'
DATA_ROOT = './experiments/cub/CUB200/class_attr_data_10/'

def main():
    # 1. Configuration for the 112-concept checkpoint
    N_CONCEPTS = 112 
    
    # This list tells the loader which 112 attributes to keep (indices 0-111)
    selected_indices = list(range(N_CONCEPTS))
    
    def concept_transform(sample_c):
        # Converts 312 attributes to the 112 the model expects
        return torch.tensor(sample_c)[selected_indices]

    # 2. Setup Data Loader with the transform
    test_loader = load_data(
        pkl_paths=[os.path.join(DATA_ROOT, 'test.pkl')],
        use_attr=True,
        no_img=True,
        batch_size=64,
        root_dir=DATA_ROOT,
        concept_transform=concept_transform # Crucial: filters 312 -> 112
    )

    # 3. Initialize Model with 112 concepts
    model = MNISTModel(
        encoder=CUBEncoder(
            input_size=N_CONCEPTS, 
            emb_size=EMB_SIZE, 
            cp_output=N_CONCEPTS
        ),
        emb_size=EMB_SIZE,
        rule_emb_size=RULE_EMB,
        n_tasks=200,
        n_rules=N_RULES,
        n_concepts=N_CONCEPTS,
        concept_names=CONCEPT_SEMANTICS[:N_CONCEPTS], # Use first 112 names
        rule_module=AECat,
        selector_input=InputTypes.embedding
    )

    # 4. Load Trained Weights with Security Override
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(
            CHECKPOINT_PATH, 
            map_location=torch.device('cpu'), 
            weights_only=False
        )
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Successfully loaded model with {N_CONCEPTS} concepts.")
    else:
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    model.eval()

    # 5. Extract and Print
    print("\nExtracting learned rules...")
    task_to_rules, _ = model.aggregate_rules(test_loader, type='most_likely')

    print("\n" + "="*50)
    print("LEARNED CONCEPT RULES (Sample)")
    print("="*50)

    for task_idx in range(10): # Printing first 10 species
        species_name = CLASS_NAMES[task_idx]
        print(f"\n[{species_name}]:")
        print(f"Rule: {task_to_rules[task_idx]}")

if __name__ == '__main__':
    main()
