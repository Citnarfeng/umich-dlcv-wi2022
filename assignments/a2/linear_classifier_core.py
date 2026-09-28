# University of Michigan EECS 498/598, Winter 2022 course implementation.
# Compact core; algorithms and public interfaces retained from the completed assignment.

import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional

def hello_linear_classifier():
    print('Hello from linear_classifier.py!')

class LinearClassifier:

    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, learning_rate: float=0.001, reg: float=1e-05, num_iters: int=100, batch_size: int=200, verbose: bool=False):
        train_args = (self.loss, self.W, X_train, y_train, learning_rate, reg, num_iters, batch_size, verbose)
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(self, W: torch.Tensor, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({'W': self.W}, path)
        print('Saved in {}'.format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location='cpu')
        self.W = W_dict['W']
        if self.W is None:
            raise Exception('Failed to load your checkpoint')

class LinearSVM(LinearClassifier):

    def loss(self, W: torch.Tensor, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)

class Softmax(LinearClassifier):

    def loss(self, W: torch.Tensor, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)

def svm_loss_naive(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    dW = torch.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = W.t().mv(X[i])
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
    loss /= num_train
    loss += reg * torch.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
    return (loss, dW)

def svm_loss_vectorized(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    loss = 0.0
    dW = torch.zeros_like(W)
    num_train = X.shape[0]
    idx = torch.arange(num_train, device=X.device)
    scores = X @ W
    margins = (scores - scores[idx, y].unsqueeze(dim=1) + 1).clamp_min(0)
    margins[idx, y] = 0
    loss = margins.sum() / num_train + reg * torch.sum(W * W)
    mask = torch.zeros_like(margins)
    mask[margins > 0] = 1
    mask[idx, y] = -mask.sum(dim=1)
    dW = X.t() @ mask / num_train + 2 * reg * W
    return (loss, dW)

def sample_batch(X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int):
    X_batch = None
    y_batch = None
    idx = torch.randint(num_train, (batch_size,), device=X.device)
    X_batch, y_batch = (X[idx], y[idx])
    return (X_batch, y_batch)

def train_linear_classifier(loss_func: Callable, W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, learning_rate: float=0.001, reg: float=1e-05, num_iters: int=100, batch_size: int=200, verbose: bool=False):
    num_train, dim = X.shape
    if W is None:
        num_classes = torch.max(y) + 1
        W = 1e-06 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)
    else:
        num_classes = W.shape[1]
    loss_history = []
    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())
        W -= learning_rate * grad
        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))
    return (W, loss_history)

def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    y_pred = (X @ W).argmax(dim=1)
    return y_pred

def svm_get_search_params():
    learning_rates = []
    regularization_strengths = []
    learning_rates = [0.001, 0.005, 0.01]
    regularization_strengths = [0.05, 0.1, 0.2]
    return (learning_rates, regularization_strengths)

def test_one_param_set(cls: LinearClassifier, data_dict: Dict[str, torch.Tensor], lr: float, reg: float, num_iters: int=2000):
    train_acc = 0.0
    val_acc = 0.0
    cls.train(data_dict['X_train'], data_dict['y_train'], learning_rate=lr, reg=reg, num_iters=num_iters)
    train_acc = 100 * (cls.predict(data_dict['X_train']) == data_dict['y_train']).float().mean().item()
    val_acc = 100 * (cls.predict(data_dict['X_val']) == data_dict['y_val']).float().mean().item()
    return (cls, train_acc, val_acc)

def softmax_loss_naive(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    loss = 0.0
    dW = torch.zeros_like(W)
    num_train, num_classes = (X.shape[0], W.shape[1])
    for i in range(num_train):
        scores = X[i] @ W
        scores -= scores.max()
        exp_scores = torch.exp(scores)
        probs = exp_scores / exp_scores.sum()
        loss += torch.log(exp_scores.sum()) - scores[y[i]]
        for j in range(num_classes):
            dW[:, j] += probs[j] * X[i]
        dW[:, y[i]] -= X[i]
    loss = loss / num_train + reg * torch.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    return (loss, dW)

def softmax_loss_vectorized(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    loss = 0.0
    dW = torch.zeros_like(W)
    num_train = X.shape[0]
    idx = torch.arange(num_train, device=X.device)
    scores = X @ W
    scores -= scores.max(dim=1, keepdim=True).values
    exp_scores = torch.exp(scores)
    sums = exp_scores.sum(dim=1, keepdim=True)
    loss = (torch.log(sums[:, 0]) - scores[idx, y]).mean() + reg * torch.sum(W * W)
    probs = exp_scores / sums
    probs[idx, y] -= 1
    dW = X.t() @ probs / num_train + 2 * reg * W
    return (loss, dW)

def softmax_get_search_params():
    learning_rates = []
    regularization_strengths = []
    learning_rates = [0.001, 0.005, 0.01, 0.02]
    regularization_strengths = [0.01, 0.05, 0.1, 0.2]
    return (learning_rates, regularization_strengths)
