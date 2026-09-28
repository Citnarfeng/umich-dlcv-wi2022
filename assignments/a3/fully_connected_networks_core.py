# University of Michigan EECS 498/598, Winter 2022 course implementation.
# Compact core; algorithms and public interfaces retained from the completed assignment.

import torch
from a3_helper_core import softmax_loss
from eecs598 import Solver

def hello_fully_connected_networks():
    print('Hello from fully_connected_networks.py!')

class Linear(object):

    @staticmethod
    def forward(x, w, b):
        out = None
        out = x.reshape(x.shape[0], -1).mm(w) + b
        cache = (x, w, b)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        x, w, b = cache
        dx, dw, db = (None, None, None)
        x_flat = x.reshape(x.shape[0], -1)
        dx = dout.mm(w.t()).reshape(x.shape)
        dw = x_flat.t().mm(dout)
        db = dout.sum(dim=0)
        return (dx, dw, db)

class ReLU(object):

    @staticmethod
    def forward(x):
        out = None
        out = torch.clamp(x, min=0)
        cache = x
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        dx, x = (None, cache)
        dx = dout * (x > 0)
        return dx

class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return (dx, dw, db)

class TwoLayerNet(object):

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10, weight_scale=0.001, reg=0.0, dtype=torch.float32, device='cpu'):
        self.params = {}
        self.reg = reg
        self.params['W1'] = weight_scale * torch.randn(input_dim, hidden_dim, dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W2'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {'reg': self.reg, 'params': self.params}
        torch.save(checkpoint, path)
        print('Saved in {}'.format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print('load checkpoint file: {}'.format(path))

    def loss(self, X, y=None):
        scores = None
        hidden, hidden_cache = Linear_ReLU.forward(X, self.params['W1'], self.params['b1'])
        scores, scores_cache = Linear.forward(hidden, self.params['W2'], self.params['b2'])
        if y is None:
            return scores
        loss, grads = (0, {})
        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (self.params['W1'].square().sum() + self.params['W2'].square().sum())
        dhidden, grads['W2'], grads['b2'] = Linear.backward(dscores, scores_cache)
        _, grads['W1'], grads['b1'] = Linear_ReLU.backward(dhidden, hidden_cache)
        grads['W1'] += 2 * self.reg * self.params['W1']
        grads['W2'] += 2 * self.reg * self.params['W2']
        return (loss, grads)

class FullyConnectedNet(object):

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10, dropout=0.0, reg=0.0, weight_scale=0.01, seed=None, dtype=torch.float, device='cpu'):
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W%d' % (i + 1)] = weight_scale * torch.randn(dims[i], dims[i + 1], dtype=dtype, device=device)
            self.params['b%d' % (i + 1)] = torch.zeros(dims[i + 1], dtype=dtype, device=device)
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {'reg': self.reg, 'dtype': self.dtype, 'params': self.params, 'num_layers': self.num_layers, 'use_dropout': self.use_dropout, 'dropout_param': self.dropout_param}
        torch.save(checkpoint, path)
        print('Saved in {}'.format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print('load checkpoint file: {}'.format(path))

    def loss(self, X, y=None):
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        caches = []
        out = X
        for i in range(1, self.num_layers):
            out, layer_cache = Linear_ReLU.forward(out, self.params['W%d' % i], self.params['b%d' % i])
            dropout_cache = None
            if self.use_dropout:
                out, dropout_cache = Dropout.forward(out, self.dropout_param)
            caches.append((layer_cache, dropout_cache))
        scores, final_cache = Linear.forward(out, self.params['W%d' % self.num_layers], self.params['b%d' % self.num_layers])
        if mode == 'test':
            return scores
        loss, grads = (0.0, {})
        loss, dout = softmax_loss(scores, y)
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * self.params['W%d' % i].square().sum()
        dout, dW, db = Linear.backward(dout, final_cache)
        grads['W%d' % self.num_layers] = dW + self.reg * self.params['W%d' % self.num_layers]
        grads['b%d' % self.num_layers] = db
        for i in range(self.num_layers - 1, 0, -1):
            layer_cache, dropout_cache = caches[i - 1]
            if self.use_dropout:
                dout = Dropout.backward(dout, dropout_cache)
            dout, dW, db = Linear_ReLU.backward(dout, layer_cache)
            grads['W%d' % i] = dW + self.reg * self.params['W%d' % i]
            grads['b%d' % i] = db
        return (loss, grads)

def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    solver = None
    solver = Solver(model, data_dict, update_rule=sgd, optim_config={'learning_rate': 0.1}, lr_decay=0.95, num_epochs=10, batch_size=100, print_every=100, device=device)
    return solver

def get_three_layer_network_params():
    weight_scale = 0.01
    learning_rate = 0.0001
    weight_scale = 0.1
    learning_rate = 0.1
    return (weight_scale, learning_rate)

def get_five_layer_network_params():
    learning_rate = 0.002
    weight_scale = 1e-05
    weight_scale = 0.1
    learning_rate = 0.15
    return (weight_scale, learning_rate)

def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 0.01)
    w -= config['learning_rate'] * dw
    return (w, config)

def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 0.01)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))
    next_w = None
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    config['velocity'] = v
    return (next_w, config)

def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 0.01)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-08)
    config.setdefault('cache', torch.zeros_like(w))
    next_w = None
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw.square()
    next_w = w - config['learning_rate'] * dw / (config['cache'].sqrt() + config['epsilon'])
    return (next_w, config)

def adam(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 0.001)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-08)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)
    next_w = None
    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dw.square()
    m_hat = config['m'] / (1 - config['beta1'] ** config['t'])
    v_hat = config['v'] / (1 - config['beta2'] ** config['t'])
    next_w = w - config['learning_rate'] * m_hat / (v_hat.sqrt() + config['epsilon'])
    return (next_w, config)

class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        p, mode = (dropout_param['p'], dropout_param['mode'])
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])
        mask = None
        out = None
        if mode == 'train':
            mask = (torch.rand_like(x) >= p) / (1 - p)
            out = x * mask
        elif mode == 'test':
            out = x
        cache = (dropout_param, mask)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        dropout_param, mask = cache
        mode = dropout_param['mode']
        dx = None
        if mode == 'train':
            dx = dout * mask
        elif mode == 'test':
            dx = dout
        return dx
