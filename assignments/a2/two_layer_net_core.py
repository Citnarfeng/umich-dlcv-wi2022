# University of Michigan EECS 498/598, Winter 2022 course implementation.
# Compact core; algorithms and public interfaces retained from the completed assignment.

import torch
import random
import statistics
from linear_classifier_core import sample_batch
from typing import Dict, List, Callable, Optional

def hello_two_layer_net():
    print('Hello from two_layer_net.py!')

class TwoLayerNet(object):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dtype: torch.dtype=torch.float32, device: str='cuda', std: float=0.0001):
        random.seed(0)
        torch.manual_seed(0)
        self.params = {}
        self.params['W1'] = std * torch.randn(input_size, hidden_size, dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(hidden_size, dtype=dtype, device=device)
        self.params['W2'] = std * torch.randn(hidden_size, output_size, dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(output_size, dtype=dtype, device=device)

    def loss(self, X: torch.Tensor, y: Optional[torch.Tensor]=None, reg: float=0.0):
        return nn_forward_backward(self.params, X, y, reg)

    def train(self, X: torch.Tensor, y: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, learning_rate: float=0.001, learning_rate_decay: float=0.95, reg: float=5e-06, num_iters: int=100, batch_size: int=200, verbose: bool=False):
        return nn_train(self.params, nn_forward_backward, nn_predict, X, y, X_val, y_val, learning_rate, learning_rate_decay, reg, num_iters, batch_size, verbose)

    def predict(self, X: torch.Tensor):
        return nn_predict(self.params, nn_forward_backward, X)

    def save(self, path: str):
        torch.save(self.params, path)
        print('Saved in {}'.format(path))

    def load(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint
        if len(self.params) != 4:
            raise Exception('Failed to load your checkpoint')
        for param in ['W1', 'b1', 'W2', 'b2']:
            if param not in self.params:
                raise Exception('Failed to load your checkpoint')

def nn_forward_pass(params: Dict[str, torch.Tensor], X: torch.Tensor):
    W1, b1 = (params['W1'], params['b1'])
    W2, b2 = (params['W2'], params['b2'])
    N, D = X.shape
    hidden = None
    scores = None
    hidden = (X @ W1 + b1).clamp_min(0)
    scores = hidden @ W2 + b2
    return (scores, hidden)

def nn_forward_backward(params: Dict[str, torch.Tensor], X: torch.Tensor, y: Optional[torch.Tensor]=None, reg: float=0.0):
    W1, b1 = (params['W1'], params['b1'])
    W2, b2 = (params['W2'], params['b2'])
    N, D = X.shape
    scores, h1 = nn_forward_pass(params, X)
    if y is None:
        return scores
    loss = None
    scores -= scores.max(dim=1, keepdim=True).values
    exp_scores = torch.exp(scores)
    probs = exp_scores / exp_scores.sum(dim=1, keepdim=True)
    idx = torch.arange(N, device=X.device)
    loss = (torch.log(exp_scores.sum(dim=1)) - scores[idx, y]).mean() + reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))
    grads = {}
    probs[idx, y] -= 1
    probs /= N
    grads['W2'] = h1.t() @ probs + 2 * reg * W2
    grads['b2'] = probs.sum(dim=0)
    dh = probs @ W2.t()
    dh[h1 == 0] = 0
    grads['W1'] = X.t() @ dh + 2 * reg * W1
    grads['b1'] = dh.sum(dim=0)
    return (loss, grads)

def nn_train(params: Dict[str, torch.Tensor], loss_func: Callable, pred_func: Callable, X: torch.Tensor, y: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, learning_rate: float=0.001, learning_rate_decay: float=0.95, reg: float=5e-06, num_iters: int=100, batch_size: int=200, verbose: bool=False):
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())
        for k in params:
            params[k] -= learning_rate * grads[k]
        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss.item()))
        if it % iterations_per_epoch == 0:
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            learning_rate *= learning_rate_decay
    return {'loss_history': loss_history, 'train_acc_history': train_acc_history, 'val_acc_history': val_acc_history}

def nn_predict(params: Dict[str, torch.Tensor], loss_func: Callable, X: torch.Tensor):
    y_pred = None
    y_pred = loss_func(params, X).argmax(dim=1)
    return y_pred

def nn_get_search_params():
    learning_rates = []
    hidden_sizes = []
    regularization_strengths = []
    learning_rate_decays = []
    learning_rates = [0.5, 1.0]
    hidden_sizes = [128, 256]
    regularization_strengths = [1e-05, 0.0001]
    learning_rate_decays = [0.95, 0.98]
    return (learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays)

def find_best_net(data_dict: Dict[str, torch.Tensor], get_param_set_fn: Callable):
    best_net = None
    best_stat = None
    best_val_acc = 0.0
    learning_rates, hidden_sizes, regs, decays = get_param_set_fn()
    X_train, y_train = (data_dict['X_train'], data_dict['y_train'])
    X_val, y_val = (data_dict['X_val'], data_dict['y_val'])
    input_size, output_size = (X_train.shape[1], int(y_train.max().item()) + 1)
    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for reg in regs:
                for decay in decays:
                    net = TwoLayerNet(input_size, hidden_size, output_size, dtype=X_train.dtype, device=X_train.device)
                    stat = net.train(X_train, y_train, X_val, y_val, learning_rate=lr, learning_rate_decay=decay, reg=reg, num_iters=3000, batch_size=1000)
                    val_acc = (net.predict(X_val) == y_val).float().mean().item()
                    if val_acc > best_val_acc:
                        best_net, best_stat, best_val_acc = (net, stat, val_acc)
    return (best_net, best_stat, best_val_acc)
