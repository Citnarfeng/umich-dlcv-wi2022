# Core implementation adapted from University of Michigan EECS 498/598 course starter code.
# Static validation only; not executed.
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

def hello_transformers():
    print('Hello from transformers.py!')

def generate_token_dict(vocab):
    token_dict = {}
    token_dict = {token: i for i, token in enumerate(vocab)}
    return token_dict

def prepocess_input_sequence(input_str: str, token_dict: dict, spc_tokens: list) -> list:
    out = []
    for token in input_str.split():
        if token in spc_tokens:
            out.append(token_dict[token])
        else:
            out.extend((token_dict[digit] for digit in token))
    return out

def scaled_dot_product_two_loop_single(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    out = None
    rows = []
    for i in range(query.shape[0]):
        scores = []
        for j in range(key.shape[0]):
            scores.append((query[i] * key[j]).sum() / query.shape[1] ** 0.5)
        rows.append(torch.softmax(torch.stack(scores), dim=0) @ value)
    out = torch.stack(rows)
    return out

def scaled_dot_product_two_loop_batch(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    out = None
    N, K, M = query.shape
    rows = []
    for i in range(K):
        scores = []
        for j in range(key.shape[1]):
            scores.append((query[:, i] * key[:, j]).sum(dim=1) / M ** 0.5)
        weights = torch.softmax(torch.stack(scores, dim=1), dim=1)
        rows.append(torch.bmm(weights.unsqueeze(1), value).squeeze(1))
    out = torch.stack(rows, dim=1)
    return out

def scaled_dot_product_no_loop_batch(query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
    _, _, M = query.shape
    y = None
    weights_softmax = None
    scores = torch.bmm(query, key.transpose(1, 2)) / M ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask, -1000000000.0)
    weights_softmax = torch.softmax(scores, dim=-1)
    y = torch.bmm(weights_softmax, value)
    return (y, weights_softmax)

class SelfAttention(nn.Module):

    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()
        self.q = None
        self.k = None
        self.v = None
        self.weights_softmax = None
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_q)
        self.v = nn.Linear(dim_in, dim_v)
        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
        self.weights_softmax = None
        y = None
        y, self.weights_softmax = scaled_dot_product_no_loop_batch(self.q(query), self.k(key), self.v(value), mask)
        return y

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * dim_out, dim_in)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
        y = None
        y = self.linear(torch.cat([head(query, key, value, mask) for head in self.heads], dim=-1))
        return y

class LayerNormalization(nn.Module):

    def __init__(self, emb_dim: int, epsilon: float=1e-10):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor):
        y = None
        centered = x - x.mean(dim=-1, keepdim=True)
        y = self.gamma * centered / torch.sqrt(centered.square().mean(dim=-1, keepdim=True) + self.epsilon) + self.beta
        return y

class FeedForwardBlock(nn.Module):

    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        self.linear2 = nn.Linear(hidden_dim_feedforward, inp_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        y = None
        y = self.linear2(F.relu(self.linear1(x)))
        return y

class EncoderBlock(nn.Module):

    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f'The value emb_dim = {emb_dim} is not divisible\n                             by num_heads = {num_heads}. Please select an\n                             appropriate value.')
        self.MultiHeadBlock = MultiHeadAttention(num_heads, emb_dim, emb_dim // num_heads)
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.feed_forward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = None
        out = self.dropout(self.norm1(x + self.MultiHeadBlock(x, x, x)))
        y = self.dropout(self.norm2(out + self.feed_forward(out)))
        return y

def get_subsequent_mask(seq):
    mask = None
    mask = torch.ones(seq.shape[1], seq.shape[1], dtype=torch.bool, device=seq.device).triu(1).unsqueeze(0).expand(seq.shape[0], -1, -1)
    return mask

class DecoderBlock(nn.Module):

    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f'The value emb_dim = {emb_dim} is not divisible\n                             by num_heads = {num_heads}. Please select an\n                             appropriate value.')
        self.attention_self = None
        self.attention_cross = None
        self.feed_forward = None
        self.norm1 = None
        self.norm2 = None
        self.norm3 = None
        self.dropout = None
        self.feed_forward = None
        self.attention_self = MultiHeadAttention(num_heads, emb_dim, emb_dim // num_heads)
        self.attention_cross = MultiHeadAttention(num_heads, emb_dim, emb_dim // num_heads)
        self.feed_forward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.norm3 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_inp: Tensor, enc_inp: Tensor, mask: Tensor=None) -> Tensor:
        y = None
        out = self.dropout(self.norm1(dec_inp + self.attention_self(dec_inp, dec_inp, dec_inp, mask)))
        out = self.dropout(self.norm2(out + self.attention_cross(out, enc_inp, enc_inp)))
        y = self.dropout(self.norm3(out + self.feed_forward(out)))
        return y

class Encoder(nn.Module):

    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout) for _ in range(num_layers)])

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)
        return src_seq

class Decoder(nn.Module):

    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, num_layers: int, dropout: float, vocab_len: int):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout) for _ in range(num_layers)])
        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
        a = (6 / (emb_dim + vocab_len)) ** 0.5
        nn.init.uniform_(self.proj_to_vocab.weight, -a, a)

    def forward(self, target_seq: Tensor, enc_out: Tensor, mask: Tensor):
        out = target_seq.clone()
        for _layer in self.layers:
            out = _layer(out, enc_out, mask)
        out = self.proj_to_vocab(out)
        return out

def position_encoding_simple(K: int, M: int) -> Tensor:
    y = None
    y = (torch.arange(K, dtype=torch.float32) / K).reshape(1, K, 1).expand(1, K, M)
    return y

def position_encoding_sinusoid(K: int, M: int) -> Tensor:
    y = None
    position = torch.arange(K, dtype=torch.float32).unsqueeze(1)
    frequency = 10000 ** torch.floor(torch.arange(0, M, 2, dtype=torch.float32) / M)
    angles = position / frequency
    y = torch.empty(1, K, M)
    y[0, :, 0::2] = torch.sin(angles)
    y[0, :, 1::2] = torch.cos(angles[:, :M // 2])
    return y

class Transformer(nn.Module):

    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float, num_enc_layers: int, num_dec_layers: int, vocab_len: int):
        super().__init__()
        self.emb_layer = None
        self.emb_layer = nn.Embedding(vocab_len, emb_dim)
        self.encoder = Encoder(num_heads, emb_dim, feedforward_dim, num_enc_layers, dropout)
        self.decoder = Decoder(num_heads, emb_dim, feedforward_dim, num_dec_layers, dropout, vocab_len)

    def forward(self, ques_b: Tensor, ques_pos: Tensor, ans_b: Tensor, ans_pos: Tensor) -> Tensor:
        q_emb = self.emb_layer(ques_b)
        a_emb = self.emb_layer(ans_b)
        q_emb_inp = q_emb + ques_pos
        a_emb_inp = a_emb[:, :-1] + ans_pos[:, :-1]
        dec_out = None
        enc_out = self.encoder(q_emb_inp)
        mask = get_subsequent_mask(ans_b[:, :-1])
        dec_out = self.decoder(a_emb_inp, enc_out, mask)
        dec_out = dec_out.reshape(-1, dec_out.shape[-1])
        return dec_out

class AddSubDataset(torch.utils.data.Dataset):

    def __init__(self, input_seqs, target_seqs, convert_str_to_tokens, special_tokens, emb_dim, pos_encode):
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.convert_str_to_tokens = convert_str_to_tokens
        self.emb_dim = emb_dim
        self.special_tokens = special_tokens
        self.pos_encode = pos_encode

    def preprocess(self, inp):
        return prepocess_input_sequence(inp, self.convert_str_to_tokens, self.special_tokens)

    def __getitem__(self, idx):
        inp = self.input_seqs[idx]
        out = self.target_seqs[idx]
        preprocess_inp = torch.tensor(self.preprocess(inp))
        preprocess_out = torch.tensor(self.preprocess(out))
        inp_pos = len(preprocess_inp)
        inp_pos_enc = self.pos_encode(inp_pos, self.emb_dim)
        out_pos = len(preprocess_out)
        out_pos_enc = self.pos_encode(out_pos, self.emb_dim)
        return (preprocess_inp, inp_pos_enc[0], preprocess_out, out_pos_enc[0])

    def __len__(self):
        return len(self.input_seqs)

def LabelSmoothingLoss(pred, ground):
    ground = ground.contiguous().view(-1)
    eps = 0.1
    n_class = pred.size(1)
    one_hot = torch.nn.functional.one_hot(ground).to(pred.dtype)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.sum()
    return loss

def CrossEntropyLoss(pred, ground):
    loss = F.cross_entropy(pred, ground, reduction='sum')
    return loss
