import torchvision
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from itertools import product
from collections import defaultdict

from pathlib import Path

data_path = str(Path(__file__).parent / "data" / "raw")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

def get_mnist_data(train: bool) -> MNIST:
    return torchvision.datasets.MNIST(root=data_path, train=train, download=True, transform=transform)

def addition_dataset(train, num_digits, digit_limit=10):
    dataset = get_mnist_data(train)
    X, y = dataset.data, dataset.targets

    X = X[y < digit_limit]
    y = y[y < digit_limit]

    X = torch.unsqueeze(X, 1).float()
    size = len(X) // num_digits
    
    X, y = torch.split(X, size), torch.split(y, size)

    if len(X) % num_digits != 0:
        X = X[:-1]
        y = y[:-1]

    c = [torch.zeros((len(X[0]), digit_limit)).float() for _ in range(len(X))]
    for i, ys in enumerate(y):
        for j, yi in enumerate(ys):
            c[i][j, yi] = 1.0

    y = torch.sum(torch.stack(y, 0), 0)
    return X, c, y

def create_single_digit_addition(num_digits, digit_limit=10):
    concept_names = ["x%d%d" % (i, j) for i, j in product(range(num_digits), range(digit_limit))]


    sums = defaultdict(list)
    for d in product(*[range(digit_limit) for _ in range(num_digits)]):
        conj = []
        z = 0
        for i, n in enumerate(d):
            conj.append("x%d%d" % (i,n))
            z += n
        sums[z].append("(" + " & ".join(conj) + ")")


    explanations = {}
    class_names = ["z%d" % z for z in range(digit_limit*num_digits - num_digits + 1)]
    for z in range(digit_limit*num_digits - num_digits + 1):

        explanations["z%d" % z] = {"name": "%d" % z,
                                   "explanation": "(" + " | ".join(sums[z]) + ")"}

    return concept_names, class_names, explanations