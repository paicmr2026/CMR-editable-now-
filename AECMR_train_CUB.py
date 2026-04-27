import torch
torch.cuda.empty_cache()
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from experiments.mnist.autoencoderCMR import (
    MNISTModel, MNISTEncoder, AECat,
    InputTypes, SimilarityTypes, SaveBestModelCallbackVal,
    get_accuracy, get_concept_accuracy
)
from experiments.cub.models import CUBEncoder
from experiments.cub.CUB200.cub_loader import CLASS_NAMES, CONCEPT_SEMANTICS, load_data

EMB_SIZE     = 100   
RULE_EMB     = 500   
N_RULES      = 3     
LR           = 0.001
BATCH_SIZE   = 64
MAX_EPOCHS   = 100
SEED         = 42

def main():
    pl.seed_everything(SEED)

    root_dir = './experiments/cub/CUB200/class_attr_data_10/'
    
    train_loader = load_data(
        pkl_paths=[os.path.join(root_dir, 'train.pkl')],
        use_attr=True,
        no_img=True,
        batch_size=BATCH_SIZE,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=None,
        resampling=False,
        root_dir=root_dir,
        num_workers=8
    )

    val_loader = load_data(
        pkl_paths=[os.path.join(root_dir, 'val.pkl')],
        use_attr=True,
        no_img=True,
        batch_size=BATCH_SIZE,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=None,
        resampling=False,
        root_dir=root_dir,
        num_workers=8
    )

    test_loader = load_data(
        pkl_paths=[os.path.join(root_dir, 'test.pkl')],
        use_attr=True,
        no_img=True,
        batch_size=BATCH_SIZE,
        uncertain_label=False,
        n_class_attr=2,
        image_dir=None,
        resampling=False,
        root_dir=root_dir,
        num_workers=8
    )

    sample_batch = next(iter(train_loader))
    sample_x, sample_c, sample_y = sample_batch
    
    model = MNISTModel(
        encoder=CUBEncoder(input_size=sample_x.shape[1], emb_size=EMB_SIZE, cp_output=sample_c.shape[1]),
        emb_size=EMB_SIZE,
        rule_emb_size=RULE_EMB,
        n_concepts=sample_c.shape[1],
        n_tasks=200, # Number of tasks in CUB
        n_rules=N_RULES,
        concept_names=CONCEPT_SEMANTICS,
        rule_module=AECat,
        selector_input=InputTypes.embedding,
        lr=LR,
        w_c=1, 
        w_y=30, 
        w_yF=0.005,
        reset_selector=True,
        reset_selector_every_n_epochs=30
    )

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
    
    trainer.fit(model, train_loader, val_loader)

    model.load_state_dict(cb.best_state_dict)
    model.eval()

    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)
    
    print(f"\nCUB Train Acc: {train_acc:.4f}")
    print(f"CUB Test Acc:  {test_acc:.4f}")

if __name__ == '__main__':
    main()