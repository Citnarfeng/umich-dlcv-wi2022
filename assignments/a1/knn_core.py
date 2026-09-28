# University of Michigan EECS 498/598, Winter 2022 course implementation.
# Compact core; algorithms and public interfaces retained from the completed assignment.

import torch
from typing import Dict, List

def hello():
    print('Hello from knn.py!')

def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    x_train_flat = x_train.flatten(1)
    x_test_flat = x_test.flatten(1)
    for i in range(num_train):
        for j in range(num_test):
            ling = x_train_flat[i] - x_test_flat[j]
            dists[i, j] = torch.dot(ling, ling)
    return dists

def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    x_train_flat = x_train.flatten(1)
    x_test_flat = x_test.flatten(1)
    test_sq = (x_test_flat * x_test_flat).sum(1)
    for i in range(num_train):
        xi = x_train_flat[i]
        dists[i] = test_sq + torch.dot(xi, xi) - 2 * (x_test_flat @ xi)
    return dists

def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    x_train_flat = x_train.flatten(1)
    x_test_flat = x_test.flatten(1)
    train_sq = (x_train_flat * x_train_flat).sum(1, keepdim=True)
    test_sq = (x_test_flat * x_test_flat).sum(1)
    torch.mm(x_train_flat, x_test_flat.T, out=dists)
    dists.mul_(-2).add_(train_sq).add_(test_sq)
    return dists

def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int=1):
    num_train, num_test = dists.shape
    y_pred = y_train.new_zeros(num_test)
    nearest = torch.topk(dists, k, dim=0, largest=False, sorted=False).indices
    for j in range(num_test):
        y_pred[j] = torch.bincount(y_train[nearest[:, j]]).argmax()
    '\n    num_classes = int(y_train.max()) + 1\n\n    nearest = torch.topk(dists, k, dim = 0, largest = False, sorted = False).indices; labels = y_train[nearest]\n    offset = torch.arange(num_test, device = y_train.device, dtype = y_train.dtype) * num_classes; labels.add_(offset)\n    votes = torch.bincount(labels.reshape(-1), minlength = num_test * num_classes).view(num_test, num_classes)\n    y_pred = votes.argmax(1)\n    '
    return y_pred

class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test: torch.Tensor, k: int=1):
        y_test_pred = None
        dists = compute_distances_no_loops(self.x_train, x_test)
        y_test_pred = predict_labels(dists, self.y_train, k)
        return y_test_pred

    def check_accuracy(self, x_test: torch.Tensor, y_test: torch.Tensor, k: int=1, quiet: bool=False):
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = f'Got {num_correct} / {num_samples} correct; accuracy is {accuracy:.2f}%'
        if not quiet:
            print(msg)
        return accuracy

def knn_cross_validate(x_train: torch.Tensor, y_train: torch.Tensor, num_folds: int=5, k_choices: List[int]=[1, 3, 5, 8, 10, 12, 15, 20, 50, 100]):
    x_train_folds = []
    y_train_folds = []
    x_train_folds = list(torch.chunk(x_train, num_folds, dim=0))
    y_train_folds = list(torch.chunk(y_train, num_folds, dim=0))
    k_to_accuracies = {}
    k_to_accuracies = {k: [] for k in k_choices}
    for i in range(num_folds):
        x_val, y_val = (x_train_folds[i], y_train_folds[i])
        x_tr = torch.cat(x_train_folds[:i] + x_train_folds[i + 1:], dim=0)
        y_tr = torch.cat(y_train_folds[:i] + y_train_folds[i + 1:], dim=0)
        dists = compute_distances_no_loops(x_tr, x_val)
        for k in k_choices:
            y_pred = predict_labels(dists, y_tr, k)
            k_to_accuracies[k].append(100.0 * (y_pred == y_val).sum().item() / y_val.numel())
    return k_to_accuracies

def knn_get_best_k(k_to_accuracies: Dict[int, List]):
    best_k = 0
    best_k = min(k_to_accuracies, key=lambda k: (-sum(k_to_accuracies[k]), k))
    return best_k
