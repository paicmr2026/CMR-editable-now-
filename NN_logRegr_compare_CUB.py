import torch
import os
import lightning.pytorch as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURATION ---
BATCH_SIZE = 64
MAX_EPOCHS = 50
SEED       = 42
LR         = 0.001

class BaselineModels(pl.LightningModule):
    def __init__(self, input_size, n_tasks, lr=1e-3, model_type='nn'):
        super().__init__()
        self.lr = lr
        
        if model_type == 'logistic':
            self.model = nn.Linear(input_size, n_tasks)
        else:
            # Simpel Neuraal Netwerk (MLP)
            self.model = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, n_tasks)
            )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def get_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, _, y in loader:
            y_hat = model(x)
            preds = torch.argmax(y_hat, dim=1)
            targets = torch.argmax(y, dim=1)
            correct += (preds == targets).sum().item()
            total += y.size(0)
    return correct / total

def run_experiment(model_type, x_train, y_train, x_test, y_test):
    print(f"\n--- Training {model_type.upper()} Baseline ---")
    train_loader = DataLoader(TensorDataset(x_train, x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(x_test, x_test, y_test), batch_size=BATCH_SIZE)

    model = BaselineModels(input_size=x_train.shape[1], n_tasks=200, lr=LR, model_type=model_type)
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="auto", devices=1, logger=False, enable_checkpointing=False)
    
    trainer.fit(model, train_loader)
    
    model.to("cpu")
    train_acc = get_accuracy(model, train_loader)
    test_acc  = get_accuracy(model, test_loader)
    print(f"{model_type.upper()} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

def main():
    pl.seed_everything(SEED)
    root = './experiments/cub/embeddings/cub/'
    
    x_train = torch.load(os.path.join(root, 'train_x.pt')).float()
    y_train = torch.nn.functional.one_hot(torch.load(os.path.join(root, 'train_y.pt')).long(), num_classes=200).float()
    x_test = torch.load(os.path.join(root, 'test_x.pt')).float()
    y_test = torch.nn.functional.one_hot(torch.load(os.path.join(root, 'test_y.pt')).long(), num_classes=200).float()

    # Run beide baselines
    run_experiment('logistic', x_train, y_train, x_test, y_test)
    run_experiment('nn', x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()