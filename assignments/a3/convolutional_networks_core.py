# University of Michigan EECS 498/598, Winter 2022 course implementation.
# Compact core; algorithms and public interfaces retained from the completed assignment.

import torch
from a3_helper_core import softmax_loss
from fully_connected_networks_core import Linear_ReLU, Linear, Solver, adam, ReLU

def hello_convolutional_networks():
    print('Hello from convolutional_networks.py!')

class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        out = None
        stride, pad = (conv_param['stride'], conv_param['pad'])
        N, _, H, W = x.shape
        F, _, HH, WW = w.shape
        H_out = 1 + (H + 2 * pad - HH) // stride
        W_out = 1 + (W + 2 * pad - WW) // stride
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        out = torch.empty((N, F, H_out, W_out), dtype=x.dtype, device=x.device)
        for i in range(H_out):
            hs = i * stride
            for j in range(W_out):
                ws = j * stride
                window = x_pad[:, :, hs:hs + HH, ws:ws + WW]
                out[:, :, i, j] = (window[:, None] * w[None]).sum((2, 3, 4)) + b
        cache = (x, w, b, conv_param)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        dx, dw, db = (None, None, None)
        x, w, _, conv_param = cache
        stride, pad = (conv_param['stride'], conv_param['pad'])
        N, _, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, H_out, W_out = dout.shape
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        dx_pad = torch.zeros_like(x_pad)
        dw = torch.zeros_like(w)
        db = dout.sum((0, 2, 3))
        for i in range(H_out):
            hs = i * stride
            for j in range(W_out):
                ws = j * stride
                window = x_pad[:, :, hs:hs + HH, ws:ws + WW]
                dw += (dout[:, :, i, j, None, None, None] * window[:, None]).sum(0)
                dx_pad[:, :, hs:hs + HH, ws:ws + WW] += (dout[:, :, i, j, None, None, None] * w[None]).sum(1)
        dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
        return (dx, dw, db)

class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        out = None
        N, C, H, W = x.shape
        ph, pw, stride = (pool_param['pool_height'], pool_param['pool_width'], pool_param['stride'])
        H_out = 1 + (H - ph) // stride
        W_out = 1 + (W - pw) // stride
        out = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
        for i in range(H_out):
            hs = i * stride
            for j in range(W_out):
                ws = j * stride
                out[:, :, i, j] = x[:, :, hs:hs + ph, ws:ws + pw].amax((2, 3))
        cache = (x, pool_param)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        dx = None
        x, pool_param = cache
        ph, pw, stride = (pool_param['pool_height'], pool_param['pool_width'], pool_param['stride'])
        dx = torch.zeros_like(x)
        for i in range(dout.shape[2]):
            hs = i * stride
            for j in range(dout.shape[3]):
                ws = j * stride
                window = x[:, :, hs:hs + ph, ws:ws + pw]
                index = window.reshape(*window.shape[:2], -1).argmax(dim=2)
                maxima = torch.arange(ph * pw, device=x.device).reshape(ph, pw) == index[..., None, None]
                dx[:, :, hs:hs + ph, ws:ws + pw] += maxima * dout[:, :, i, j, None, None]
        return dx

class ThreeLayerConvNet(object):

    def __init__(self, input_dims=(3, 32, 32), num_filters=32, filter_size=7, hidden_dim=100, num_classes=10, weight_scale=0.001, reg=0.0, dtype=torch.float, device='cpu'):
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dims
        pooled_dim = num_filters * (H // 2) * (W // 2)
        self.params['W1'] = weight_scale * torch.randn(num_filters, C, filter_size, filter_size, dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(num_filters, dtype=dtype, device=device)
        self.params['W2'] = weight_scale * torch.randn(pooled_dim, hidden_dim, dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W3'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {'reg': self.reg, 'dtype': self.dtype, 'params': self.params}
        torch.save(checkpoint, path)
        print('Saved in {}'.format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print('load checkpoint file: {}'.format(path))

    def loss(self, X, y=None):
        X = X.to(self.dtype)
        W1, b1 = (self.params['W1'], self.params['b1'])
        W2, b2 = (self.params['W2'], self.params['b2'])
        W3, b3 = (self.params['W3'], self.params['b3'])
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        scores = None
        out1, cache1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out2, cache2 = Linear_ReLU.forward(out1, W2, b2)
        scores, cache3 = Linear.forward(out2, W3, b3)
        if y is None:
            return scores
        loss, grads = (0.0, {})
        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (W1.square().sum() + W2.square().sum() + W3.square().sum())
        dout2, grads['W3'], grads['b3'] = Linear.backward(dscores, cache3)
        dout1, grads['W2'], grads['b2'] = Linear_ReLU.backward(dout2, cache2)
        _, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dout1, cache1)
        for i in range(1, 4):
            grads['W%d' % i] += 2 * self.reg * self.params['W%d' % i]
        return (loss, grads)

class DeepConvNet(object):

    def __init__(self, input_dims=(3, 32, 32), num_filters=[8, 8, 8, 8, 8], max_pools=[0, 1, 2, 3, 4], batchnorm=False, num_classes=10, weight_scale=0.001, reg=0.0, weight_initializer=None, dtype=torch.float, device='cpu'):
        self.params = {}
        self.num_layers = len(num_filters) + 1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype
        if device == 'cuda':
            device = 'cuda:0'
        C, H, W = input_dims
        for i, F in enumerate(num_filters, 1):
            if weight_scale == 'kaiming':
                W_i = kaiming_initializer(C, F, K=3, device=device, dtype=dtype)
            elif weight_initializer is not None:
                W_i = weight_initializer(C, F, K=3, device=device, dtype=dtype)
            else:
                W_i = weight_scale * torch.randn(F, C, 3, 3, dtype=dtype, device=device)
            self.params['W%d' % i] = W_i
            self.params['b%d' % i] = torch.zeros(F, dtype=dtype, device=device)
            if batchnorm:
                self.params['gamma%d' % i] = torch.ones(F, dtype=dtype, device=device)
                self.params['beta%d' % i] = torch.zeros(F, dtype=dtype, device=device)
            C = F
            if i - 1 in max_pools:
                H //= 2
                W //= 2
        D = C * H * W
        if weight_scale == 'kaiming':
            final_W = kaiming_initializer(D, num_classes, relu=False, device=device, dtype=dtype)
        elif weight_initializer is not None:
            final_W = weight_initializer(D, num_classes, relu=False, device=device, dtype=dtype)
        else:
            final_W = weight_scale * torch.randn(D, num_classes, dtype=dtype, device=device)
        self.params['W%d' % self.num_layers] = final_W
        self.params['b%d' % self.num_layers] = torch.zeros(num_classes, dtype=dtype, device=device)
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
        if not self.batchnorm:
            params_per_macro_layer = 2
        else:
            params_per_macro_layer = 4
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {'reg': self.reg, 'dtype': self.dtype, 'params': self.params, 'num_layers': self.num_layers, 'max_pools': self.max_pools, 'batchnorm': self.batchnorm, 'bn_params': self.bn_params}
        torch.save(checkpoint, path)
        print('Saved in {}'.format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        for i in range(len(self.bn_params)):
            for p in ['running_mean', 'running_var']:
                self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)
        print('load checkpoint file: {}'.format(path))

    def loss(self, X, y=None):
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        scores = None
        caches = []
        out = X
        for i in range(1, self.num_layers):
            pool = i - 1 in self.max_pools
            if self.batchnorm:
                args = (out, self.params['W%d' % i], self.params['b%d' % i], self.params['gamma%d' % i], self.params['beta%d' % i], conv_param, self.bn_params[i - 1])
                if pool:
                    out, cache = Conv_BatchNorm_ReLU_Pool.forward(*args, pool_param)
                else:
                    out, cache = Conv_BatchNorm_ReLU.forward(*args)
            elif pool:
                out, cache = Conv_ReLU_Pool.forward(out, self.params['W%d' % i], self.params['b%d' % i], conv_param, pool_param)
            else:
                out, cache = Conv_ReLU.forward(out, self.params['W%d' % i], self.params['b%d' % i], conv_param)
            caches.append(cache)
        scores, final_cache = Linear.forward(out, self.params['W%d' % self.num_layers], self.params['b%d' % self.num_layers])
        if y is None:
            return scores
        loss, grads = (0, {})
        loss, dout = softmax_loss(scores, y)
        for i in range(1, self.num_layers + 1):
            loss += self.reg * self.params['W%d' % i].square().sum()
        dout, dW, db = Linear.backward(dout, final_cache)
        grads['W%d' % self.num_layers] = dW + 2 * self.reg * self.params['W%d' % self.num_layers]
        grads['b%d' % self.num_layers] = db
        for i in range(self.num_layers - 1, 0, -1):
            pool = i - 1 in self.max_pools
            if self.batchnorm:
                if pool:
                    dout, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dout, caches[i - 1])
                else:
                    dout, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dout, caches[i - 1])
                grads['gamma%d' % i], grads['beta%d' % i] = (dgamma, dbeta)
            elif pool:
                dout, dW, db = Conv_ReLU_Pool.backward(dout, caches[i - 1])
            else:
                dout, dW, db = Conv_ReLU.backward(dout, caches[i - 1])
            grads['W%d' % i] = dW + 2 * self.reg * self.params['W%d' % i]
            grads['b%d' % i] = db
        return (loss, grads)

def find_overfit_parameters():
    weight_scale = 0.002
    learning_rate = 1e-05
    weight_scale = 0.1
    learning_rate = 0.003
    return (weight_scale, learning_rate)

def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    model = DeepConvNet(input_dims=data_dict['X_train'].shape[1:], num_filters=[32, 64, 128], max_pools=[0, 1, 2], batchnorm=True, weight_scale='kaiming', dtype=dtype, device=device)
    solver = Solver(model, data_dict, update_rule=adam, optim_config={'learning_rate': 0.001}, lr_decay=0.95, num_epochs=10, batch_size=128, print_every=100, device=device)
    return solver

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu', dtype=torch.float32):
    gain = 2.0 if relu else 1.0
    weight = None
    if K is None:
        weight = torch.randn(Din, Dout, dtype=dtype, device=device) * (gain / Din) ** 0.5
    else:
        weight = torch.randn(Dout, Din, K, K, dtype=dtype, device=device) * (gain / (Din * K * K)) ** 0.5
    return weight

class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-05)
        momentum = bn_param.get('momentum', 0.9)
        N, D = x.shape
        running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))
        out, cache = (None, None)
        if mode == 'train':
            mean = x.mean(dim=0)
            centered = x - mean
            var = centered.square().mean(dim=0)
            inv_std = torch.rsqrt(var + eps)
            x_hat = centered * inv_std
            out = gamma * x_hat + beta
            cache = (mode, centered, inv_std, x_hat, gamma)
            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var
        elif mode == 'test':
            centered = x - running_mean
            inv_std = torch.rsqrt(running_var + eps)
            x_hat = centered * inv_std
            out = gamma * x_hat + beta
            cache = (mode, centered, inv_std, x_hat, gamma)
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        dx, dgamma, dbeta = (None, None, None)
        mode, centered, inv_std, x_hat, gamma = cache
        N = dout.shape[0]
        dbeta = dout.sum(dim=0)
        dgamma = (dout * x_hat).sum(dim=0)
        dxhat = dout * gamma
        if mode == 'test':
            return (dxhat * inv_std, dgamma, dbeta)
        dvar = (dxhat * centered * -0.5 * inv_std ** 3).sum(dim=0)
        dmean = (-dxhat * inv_std).sum(dim=0) + dvar * (-2 * centered).mean(dim=0)
        dx = dxhat * inv_std + dvar * 2 * centered / N + dmean / N
        return (dx, dgamma, dbeta)

    @staticmethod
    def backward_alt(dout, cache):
        dx, dgamma, dbeta = (None, None, None)
        mode, centered, inv_std, x_hat, gamma = cache
        dbeta = dout.sum(dim=0)
        dgamma = (dout * x_hat).sum(dim=0)
        dxhat = dout * gamma
        if mode == 'test':
            return (dxhat * inv_std, dgamma, dbeta)
        dx = inv_std * (dxhat - dxhat.mean(dim=0) - x_hat * (dxhat * x_hat).mean(dim=0))
        return (dx, dgamma, dbeta)

class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        out, cache = (None, None)
        N, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        out, cache = BatchNorm.forward(flat, gamma, beta, bn_param)
        out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        dx, dgamma, dbeta = (None, None, None)
        N, C, H, W = dout.shape
        flat = dout.permute(0, 2, 3, 1).reshape(-1, C)
        dx, dgamma, dbeta = BatchNorm.backward(flat, cache)
        dx = dx.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return (dx, dgamma, dbeta)

class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = (conv_param['stride'], conv_param['pad'])
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = (torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias))
        return (dx, dw, db)

class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = (pool_param['pool_height'], pool_param['pool_width'])
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx

class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return (dx, dw, db)

class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return (dx, dw, db)

class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return (dx, dw, db, dgamma, dbeta)

class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return (dx, dw, db, dgamma, dbeta)

class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return (out, cache)

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return (dx, dw, db, dgamma, dbeta)
