"""
Biased MNIST Addition Dataset
=============================

This script creates a biased MNIST addition dataset where:

TRAINING:
    - Any sample whose digit sum == 12 is colored RED
    - All other samples remain GRAY

TESTING:
    - Random samples are colored RED regardless of sum

This creates a spurious correlation:
    RED -> SUM == 12

during training, which disappears during testing.

The script also:
    - adds RED as an explicit concept
    - visualizes samples
    - prints dataset statistics
    - tests DataLoader output shapes
"""

from pathlib import Path
from itertools import product
from collections import defaultdict

import torch
import torchvision
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_root = Path("./data")


# ============================================================
# LOAD MNIST
# ============================================================

def get_mnist_data(train: bool) -> MNIST:
    return torchvision.datasets.MNIST(
        root=str(data_root),
        train=train,
        download=True,
        transform=transform
    )


# ============================================================
# COLORIZATION
# ============================================================

def colorize_tensor(images, red_mask):
    """
    Convert grayscale MNIST tensors into RGB tensors.

    Args:
        images:
            Tensor (N,1,H,W)

        red_mask:
            Bool tensor (N,)
            True -> RED
            False -> GRAY

    Returns:
        Tensor (N,3,H,W)
    """

    N, _, H, W = images.shape

    rgb = torch.zeros((N, 3, H, W), dtype=images.dtype)

    # Default GRAY
    rgb[:, 0] = images[:, 0]
    rgb[:, 1] = images[:, 0]
    rgb[:, 2] = images[:, 0]

    # RED samples
    rgb[red_mask, 1] = 0
    rgb[red_mask, 2] = 0

    return rgb


def apply_color_bias(X, y, train=True, test_red_prob=0.5):
    """
    Apply spurious color bias.

    TRAIN:
        sum == 12 --> RED

    TEST:
        random RED
    """

    N = len(y)

    if train:
        red_mask = (y == 12)

    else:
        red_mask = torch.rand(N) < test_red_prob

    X_colored = []

    for digit_tensor in X:

        rgb_tensor = colorize_tensor(digit_tensor, red_mask)

        X_colored.append(rgb_tensor)

    return tuple(X_colored), red_mask


# ============================================================
# ADDITION DATASET
# ============================================================

def addition_dataset_red(
    train,
    num_digits,
    digit_limit=10,
    test_red_prob=0.5
):
    """
    Creates MNIST addition dataset.

    Returns:
        X           tuple of RGB digit tensors
        c           concept tensors
        y           sum labels
        red_mask    RED concept
    """

    dataset = get_mnist_data(train)

    X, y = dataset.data, dataset.targets

    # Restrict digits
    X = X[y < digit_limit]
    y = y[y < digit_limit]

    # Shape: (N,1,28,28)
    X = torch.unsqueeze(X, 1).float() / 255.0

    size = len(X) // num_digits

    X = torch.split(X, size)
    y = torch.split(y, size)

    if len(X) % num_digits != 0:
        X = X[:-1]
        y = y[:-1]

    # ========================================================
    # CONCEPT VECTORS
    # ========================================================

    c = [
        torch.zeros((len(X[0]), digit_limit)).float()
        for _ in range(len(X))
    ]

    for i, ys in enumerate(y):
        for j, yi in enumerate(ys):
            c[i][j, yi] = 1.0

    # ========================================================
    # SUM LABELS
    # ========================================================

    y = torch.sum(torch.stack(y, 0), 0)

    # ========================================================
    # APPLY COLOR BIAS
    # ========================================================

    X, red_mask = apply_color_bias(
        X,
        y,
        train=train,
        test_red_prob=test_red_prob
    )

    # ========================================================
    # OPTIONAL: ADD RED AS EXPLICIT CONCEPT
    # ========================================================

    red_concept = red_mask.float().unsqueeze(1)

    c.append(red_concept)

    return X, c, y, red_mask


# ============================================================
# LOGIC EXPLANATIONS
# ============================================================

def create_single_digit_addition(num_digits, digit_limit=10):

    concept_names = [
        "x%d%d" % (i, j)
        for i, j in product(range(num_digits), range(digit_limit))
    ]

    concept_names.append("RED")

    sums = defaultdict(list)

    for d in product(*[range(digit_limit) for _ in range(num_digits)]):

        conj = []
        z = 0

        for i, n in enumerate(d):
            conj.append("x%d%d" % (i, n))
            z += n

        sums[z].append("(" + " & ".join(conj) + ")")

    explanations = {}

    class_names = [
        "z%d" % z
        for z in range(digit_limit * num_digits - num_digits + 1)
    ]

    for z in range(digit_limit * num_digits - num_digits + 1):

        explanations["z%d" % z] = {
            "name": "%d" % z,
            "explanation": "(" + " | ".join(sums[z]) + ")"
        }

    return concept_names, class_names, explanations


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_samples(
    X,
    y,
    red_mask,
    num_digits,
    num_samples=8,
    title="samples.png"
):
    """
    Visualize dataset examples.
    """

    fig, axes = plt.subplots(
        num_samples,
        num_digits,
        figsize=(2 * num_digits, 2 * num_samples)
    )

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):

        for d in range(num_digits):

            img = X[d][i]

            # Convert CHW -> HWC
            img = img.permute(1, 2, 0).numpy()

            axes[i][d].imshow(img)
            axes[i][d].axis("off")

            if d == 0:
                axes[i][d].set_title(
                    f"sum={y[i].item()} | red={bool(red_mask[i])}"
                )

    plt.tight_layout()
    plt.savefig(title)
    print(f"\nSaved visualization to: {title}")

    plt.close()


# ============================================================
# DATASET STATISTICS
# ============================================================

def print_statistics(y, red_mask):

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total_red = red_mask.sum().item()

    print(f"Total samples: {len(y)}")
    print(f"Red samples:   {total_red}")
    print(f"Percent red:   {100 * total_red / len(y):.2f}%")

    print("\nPer-sum statistics:")

    for s in sorted(torch.unique(y).tolist()):

        mask = (y == s)

        total = mask.sum().item()

        red = red_mask[mask].sum().item()

        pct = 100 * red / total

        print(
            f"Sum={s:2d} | "
            f"total={total:5d} | "
            f"red={red:5d} | "
            f"{pct:6.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    number_digits = 2

    print("\nCreating symbolic explanations...\n")

    concept_names, class_names, explanations = \
        create_single_digit_addition(number_digits)

    print("Example concepts:")
    print(concept_names[:10])
    print("...")
    print("Last concept:", concept_names[-1])

    # ========================================================
    # TRAIN DATASET
    # ========================================================

    print("\nCreating TRAIN dataset...")

    X_train, c_train, y_train, red_train = addition_dataset_red(
        train=True,
        num_digits=number_digits,
        test_red_prob=0.5
    )

    print_statistics(y_train, red_train)

    # ========================================================
    # TEST DATASET
    # ========================================================

    print("\nCreating TEST dataset...")

    X_test, c_test, y_test, red_test = addition_dataset_red(
        train=False,
        num_digits=number_digits,
        test_red_prob=0.5
    )

    print_statistics(y_test, red_test)

    # ========================================================
    # VISUALIZE
    # ========================================================

    visualize_samples(
        X_train,
        y_train,
        red_train,
        number_digits,
        num_samples=30,
        title="train_samples.png"
    )

    visualize_samples(
        X_test,
        y_test,
        red_test,
        number_digits,
        num_samples=8,
        title="test_samples.png"
    )

    # ========================================================
    # TEST DATALOADER
    # ========================================================

    print("\nTesting DataLoader...\n")

    dataset = TensorDataset(*X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    for batch_idx, batch in enumerate(loader):

        X_batch = batch[:-1]
        y_batch = batch[-1]

        print(f"Batch {batch_idx}")

        for i, xb in enumerate(X_batch):
            print(f"Digit {i} tensor shape: {xb.shape}")

        print("Labels shape:", y_batch.shape)

        print("Example labels:")
        print(y_batch[:10])

        break

    print("\nDONE!")