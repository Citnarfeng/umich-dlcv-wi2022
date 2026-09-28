# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
import math
from typing import Optional, Tuple
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction

def hello_rnn_lstm_captioning():
    print('Hello from rnn_lstm_captioning.py!')

class ImageEncoder(nn.Module):

    def __init__(self, pretrained: bool=True, verbose: bool=True):
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)
        self.backbone = feature_extraction.create_feature_extractor(self.cnn, return_nodes={'trunk_output.block4': 'c5'})
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))['c5']
        self._out_channels = dummy_out.shape[1]
        if verbose:
            print('For input images in NCHW format, shape (2, 3, 224, 224)')
            print(f'Shape of output c5 features: {dummy_out.shape}')
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, images: torch.Tensor):
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0
        images = self.normalize(images)
        features = self.backbone(images)['c5']
        return features

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h, cache = (None, None)
    next_h = torch.tanh(x @ Wx + prev_h @ Wh + b)
    cache = (x, prev_h, Wx, Wh, next_h)
    return (next_h, cache)

def rnn_step_backward(dnext_h, cache):
    dx, dprev_h, dWx, dWh, db = (None, None, None, None, None)
    x, prev_h, Wx, Wh, next_h = cache
    da = dnext_h * (1 - next_h.square())
    dx, dprev_h = (da @ Wx.T, da @ Wh.T)
    dWx, dWh, db = (x.T @ da, prev_h.T @ da, da.sum(dim=0))
    return (dx, dprev_h, dWx, dWh, db)

def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = (None, None)
    states, cache = ([], [])
    prev_h = h0
    for t in range(x.shape[1]):
        prev_h, step_cache = rnn_step_forward(x[:, t], prev_h, Wx, Wh, b)
        states.append(prev_h)
        cache.append(step_cache)
    h = torch.stack(states, dim=1)
    return (h, cache)

def rnn_backward(dh, cache):
    dx, dh0, dWx, dWh, db = (None, None, None, None, None)
    x0, h0, Wx, Wh, _ = cache[0]
    dx = x0.new_zeros(dh.shape[0], dh.shape[1], x0.shape[1])
    dh0 = torch.zeros_like(h0)
    dWx, dWh = (torch.zeros_like(Wx), torch.zeros_like(Wh))
    db = dh.new_zeros(dh.shape[2])
    for t in reversed(range(dh.shape[1])):
        dx[:, t], dh0, dwx, dwh, db_t = rnn_step_backward(dh[:, t] + dh0, cache[t])
        dWx += dwx
        dWh += dwh
        db += db_t
    return (dx, dh0, dWx, dWh, db)

class RNN(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Wx = nn.Parameter(torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim)))
        self.Wh = nn.Parameter(torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim)))
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h

class WordEmbedding(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.W_embed = nn.Parameter(torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size)))

    def forward(self, x):
        out = None
        out = self.W_embed[x]
        return out

def temporal_softmax_loss(x, y, ignore_index=None):
    loss = None
    loss = F.cross_entropy(x.transpose(1, 2), y, ignore_index=-100 if ignore_index is None else ignore_index, reduction='sum') / x.shape[0]
    return loss

class CaptioningRNN(nn.Module):

    def __init__(self, word_to_idx, input_dim: int=512, wordvec_dim: int=128, hidden_dim: int=128, cell_type: str='rnn', image_encoder_pretrained: bool=True, ignore_index: Optional[int]=None):
        super().__init__()
        if cell_type not in {'rnn', 'lstm', 'attn'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)
        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        vocab_size = len(word_to_idx)
        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        self.ignore_index = ignore_index
        self.image_encoder = ImageEncoder(pretrained=image_encoder_pretrained)
        self.word_embed = WordEmbedding(vocab_size, wordvec_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        if cell_type == 'attn':
            self.feature_proj = nn.Conv2d(self.image_encoder.out_channels, hidden_dim, 1)
            self.rnn = AttentionLSTM(wordvec_dim, hidden_dim)
        else:
            self.feature_proj = nn.Linear(self.image_encoder.out_channels, hidden_dim)
            self.rnn = RNN(wordvec_dim, hidden_dim) if cell_type == 'rnn' else LSTM(wordvec_dim, hidden_dim)

    def forward(self, images, captions):
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        loss = 0.0
        features = self.image_encoder(images)
        initial = self.feature_proj(features if self.cell_type == 'attn' else features.mean(dim=(2, 3)))
        hidden = self.rnn(self.word_embed(captions_in), initial)
        loss = temporal_softmax_loss(self.output_proj(hidden), captions_out, self.ignore_index)
        return loss

    def sample(self, images, max_length=15):
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()
        if self.cell_type == 'attn':
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()
        with torch.no_grad():
            features = self.image_encoder(images)
            if self.cell_type == 'attn':
                A = self.feature_proj(features)
                prev_h = A.mean(dim=(2, 3))
                prev_c = prev_h
                attn_weights_all = A.new_zeros(N, max_length, A.shape[2], A.shape[3])
            else:
                prev_h = self.feature_proj(features.mean(dim=(2, 3)))
                prev_c = torch.zeros_like(prev_h)
            word = captions.new_full((N,), self._start)
            for t in range(max_length):
                embedded = self.word_embed(word)
                if self.cell_type == 'rnn':
                    prev_h = self.rnn.step_forward(embedded, prev_h)
                elif self.cell_type == 'lstm':
                    prev_h, prev_c = self.rnn.step_forward(embedded, prev_h, prev_c)
                else:
                    attn, weights = dot_product_attention(prev_h, A)
                    prev_h, prev_c = self.rnn.step_forward(embedded, prev_h, prev_c, attn)
                    attn_weights_all[:, t] = weights
                word = self.output_proj(prev_h).argmax(dim=1)
                captions[:, t] = word
        if self.cell_type == 'attn':
            return (captions, attn_weights_all.cpu())
        else:
            return captions

class LSTM(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Wx = nn.Parameter(torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim)))
        self.Wh = nn.Parameter(torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim)))
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_h, next_c = (None, None)
        i, f, o, g = (x @ self.Wx + prev_h @ self.Wh + self.b).chunk(4, dim=1)
        next_c = torch.sigmoid(f) * prev_c + torch.sigmoid(i) * torch.tanh(g)
        next_h = torch.sigmoid(o) * torch.tanh(next_c)
        return (next_h, next_c)

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        c0 = torch.zeros_like(h0)
        hn = None
        states = []
        prev_h, prev_c = (h0, c0)
        for t in range(x.shape[1]):
            prev_h, prev_c = self.step_forward(x[:, t], prev_h, prev_c)
            states.append(prev_h)
        hn = torch.stack(states, dim=1)
        return hn

def dot_product_attention(prev_h, A):
    N, H, D_a, _ = A.shape
    attn, attn_weights = (None, None)
    flat = A.flatten(2)
    weights = torch.softmax(torch.bmm(prev_h.unsqueeze(1), flat).squeeze(1) / math.sqrt(H), dim=1)
    attn = torch.bmm(flat, weights.unsqueeze(2)).squeeze(2)
    attn_weights = weights.reshape(N, A.shape[2], A.shape[3])
    return (attn, attn_weights)

class AttentionLSTM(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Wx = nn.Parameter(torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim)))
        self.Wh = nn.Parameter(torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim)))
        self.Wattn = nn.Parameter(torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim)))
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor, attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_h, next_c = (None, None)
        i, f, o, g = (x @ self.Wx + prev_h @ self.Wh + attn @ self.Wattn + self.b).chunk(4, dim=1)
        next_c = torch.sigmoid(f) * prev_c + torch.sigmoid(i) * torch.tanh(g)
        next_h = torch.sigmoid(o) * torch.tanh(next_c)
        return (next_h, next_c)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        h0 = A.mean(dim=(2, 3))
        c0 = h0
        hn = None
        states = []
        prev_h, prev_c = (h0, c0)
        for t in range(x.shape[1]):
            attn, _ = dot_product_attention(prev_h, A)
            prev_h, prev_c = self.step_forward(x[:, t], prev_h, prev_c, attn)
            states.append(prev_h)
        hn = torch.stack(states, dim=1)
        return hn
