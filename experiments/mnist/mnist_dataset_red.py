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


def apply_color_bias(
    X,
    y,
    train=True,
    train_red_prob=1.0,
    test_red_prob=0.5
):
    """
    Apply spurious color bias.

    TRAIN:
        sum == 12 --> RED with probability train_red_prob

    TEST:
        random RED with probability test_red_prob
    """

    N = len(y)

    if train:

        # Candidates
        sum12_mask = (y == 12)

        # Random probability mask
        random_mask = (
            torch.rand(N) < train_red_prob
        )

        # Only some SUM=12 become RED
        red_mask = sum12_mask & random_mask

    else:

        # Random RED during testing
        red_mask = (
            torch.rand(N) < test_red_prob
        )

    X_colored = []

    for digit_tensor in X:

        rgb_tensor = colorize_tensor(
            digit_tensor,
            red_mask
        )

        X_colored.append(rgb_tensor)

    return tuple(X_colored), red_mask


# ============================================================
# ADDITION DATASET
# ============================================================

def addition_dataset_red(
    train,
    num_digits,
    digit_limit=10,
    train_red_prob=1.0,
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

    # +1 for RED concept
    c = [
        torch.zeros((len(X[0]), digit_limit + 1)).float()
        for _ in range(len(X))
    ]

    for i, ys in enumerate(y):
        for j, yi in enumerate(ys):

            # One-hot digit concept
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
        train_red_prob=train_red_prob,
        test_red_prob=test_red_prob
    )

    # ========================================================
    # ADD RED CONCEPT TO EACH DIGIT
    # ========================================================

    for i in range(len(c)):

        # Last column = RED concept
        c[i][:, -1] = red_mask.float()

    return X, c, y, red_mask


# ============================================================
# LOGIC EXPLANATIONS
# ============================================================

def create_single_digit_addition(
    num_digits,
    digit_limit=10
):

    concept_names = [
        "x%d%d" % (i, j)
        for i, j in product(
            range(num_digits),
            range(digit_limit)
        )
    ]

    # Add RED concept per digit
    concept_names.extend([
        f"x{i}RED"
        for i in range(num_digits)
    ])

    sums = defaultdict(list)

    for d in product(
        *[range(digit_limit)
          for _ in range(num_digits)]
    ):

        conj = []
        z = 0

        for i, n in enumerate(d):

            conj.append(
                "x%d%d" % (i, n)
            )

            z += n

        sums[z].append(
            "(" + " & ".join(conj) + ")"
        )

    explanations = {}

    class_names = [
        "z%d" % z
        for z in range(
            digit_limit * num_digits
            - num_digits + 1
        )
    ]

    for z in range(
        digit_limit * num_digits
        - num_digits + 1
    ):

        explanations["z%d" % z] = {
            "name": "%d" % z,
            "explanation":
                "(" + " | ".join(sums[z]) + ")"
        }

    return (
        concept_names,
        class_names,
        explanations
    )

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
    train_red_prob=0.8,   # 80% of SUM=12 become red
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
    train_red_prob=0.8,
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

    # ========================================================
# TEST CONCEPT STRUCTURE
# ========================================================

print("\nTesting concept tensors...\n")

for digit_idx, concept_tensor in enumerate(c_train):

    print(f"Digit {digit_idx}")
    print("Shape:", concept_tensor.shape)

    # Expected shape: (N, digit_limit + 1)
    assert concept_tensor.shape[1] == 11, \
        "Expected 10 digit concepts + 1 red concept"

    # ----------------------------------------------------
    # Check one-hot encoding
    # ----------------------------------------------------

    digit_part = concept_tensor[:, :-1]

    one_hot_sums = digit_part.sum(dim=1)

    assert torch.all(one_hot_sums == 1), \
        "Digit concepts are not valid one-hot vectors"

    print("✓ One-hot encoding correct")

    # ----------------------------------------------------
    # Check red concept
    # ----------------------------------------------------

    red_column = concept_tensor[:, -1]

    assert torch.all(red_column == red_train.float()), \
        "Red concept column does not match red_mask"

    print("✓ Red concept matches red_mask")

    # ----------------------------------------------------
    # Show examples
    # ----------------------------------------------------

    print("\nExample concept vectors:")

    for i in range(30):

        digit_class = torch.argmax(
            concept_tensor[i, :-1]
        ).item()

        is_red = bool(concept_tensor[i, -1].item())

        print(
            f"Sample {i}: "
            f"digit={digit_class}, "
            f"red={is_red}"
        )

        print(concept_tensor[i])

    print("\n" + "-" * 50)

    print("\nDONE!")