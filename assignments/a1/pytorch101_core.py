# University of Michigan EECS 498/598, Winter 2022 course implementation.
# Compact core; algorithms and public interfaces retained from the completed assignment.

import torch
from typing import List, Tuple
from torch import Tensor

def hello():
    print('Hello from pytorch101.py!')

def create_sample_tensor() -> Tensor:
    x = None
    x = torch.tensor([[0, 10], [100, 0], [0, 0]])
    return x

def mutate_tensor(x: Tensor, indices: List[Tuple[int, int]], values: List[float]) -> Tensor:
    for (i, j), value in zip(indices, values):
        x[i, j] = value
    return x
import math

def count_tensor_elements(x: Tensor) -> int:
    num_elements = None
    num_elements = math.prod(x.shape)
    return num_elements

def create_tensor_of_pi(M: int, N: int) -> Tensor:
    x = None
    x = torch.full((M, N), 3.14)
    return x

def multiples_of_ten(start: int, stop: int) -> Tensor:
    assert start <= stop
    x = None
    x = 10 * torch.arange((start + 9) // 10, stop // 10 + 1, dtype=torch.float64)
    return x

def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5
    last_row = None
    third_col = None
    first_two_rows_three_cols = None
    even_rows_odd_cols = None
    last_row = x[-1, :]
    third_col = x[:, 2:3]
    first_two_rows_three_cols = x[:2, :3]
    even_rows_odd_cols = x[::2, 1::2]
    out = (last_row, third_col, first_two_rows_three_cols, even_rows_odd_cols)
    return out

def slice_assignment_practice(x: Tensor) -> Tensor:
    x[:2, 0:1] = 0
    x[:2, 1:2] = 1
    x[:2, 2:6] = 2
    x[2:4, 0:4:2] = 3
    x[2:4, 1:4:2] = 4
    x[2:4, 4:6] = 5
    return x

def shuffle_cols(x: Tensor) -> Tensor:
    y = None
    y = x[:, [0, 0, 2, 1]]
    return y

def reverse_rows(x: Tensor) -> Tensor:
    y = None
    idx = torch.arange(x.shape[0] - 1, -1, -1)
    y = x[idx]
    return y

def take_one_elem_per_col(x: Tensor) -> Tensor:
    y = None
    rows = torch.tensor([1, 0, 3], device=x.device)
    cols = torch.tensor([0, 1, 2], device=x.device)
    y = x[rows, cols]
    return y

def make_one_hot(x: List[int]) -> Tensor:
    y = None
    idx = torch.tensor(x, dtype=torch.long)
    y = torch.zeros((len(x), max(x) + 1), dtype=torch.float32)
    y[torch.arange(len(x)), idx] = 1
    return y

def sum_positive_entries(x: Tensor) -> Tensor:
    pos_sum = None
    pos_sum = x[x > 0].sum().item()
    return pos_sum

def reshape_practice(x: Tensor) -> Tensor:
    y = None
    y = x.view(2, 3, 4).transpose(0, 1).reshape(3, 8)
    return y

def zero_row_min(x: Tensor) -> Tensor:
    y = None
    y = x.clone()
    y[torch.arange(x.shape[0], device=x.device), x.argmin(dim=1)] = 0
    return y

def batched_matrix_multiply(x: Tensor, y: Tensor, use_loop: bool=True) -> Tensor:
    if use_loop:
        return batched_matrix_multiply_loop(x, y)
    else:
        return batched_matrix_multiply_noloop(x, y)

def batched_matrix_multiply_loop(x: Tensor, y: Tensor) -> Tensor:
    z = None
    B, N, _ = x.shape
    P = y.shape[2]
    z = x.new_zeros((B, N, P))
    for i in range(B):
        z[i] = x[i].mm(y[i])
    return z

def batched_matrix_multiply_noloop(x: Tensor, y: Tensor) -> Tensor:
    z = None
    z = torch.bmm(x, y)
    return z

def normalize_columns(x: Tensor) -> Tensor:
    y = None
    M = x.shape[0]
    centered = x - x.sum(dim=0) / M
    std = torch.sqrt((centered ** 2).sum(dim=0) / (M - 1))
    y = centered / std
    return y

def mm_on_cpu(x: Tensor, w: Tensor) -> Tensor:
    y = x.mm(w)
    return y

def mm_on_gpu(x: Tensor, w: Tensor) -> Tensor:
    y = None
    x_gpu = x.cuda()
    w_gpu = w.cuda()
    y = x_gpu.mm(w_gpu).cpu()
    return y

def challenge_mean_tensors(xs: List[Tensor], ls: Tensor) -> Tensor:
    y = None
    flat = torch.cat(xs)
    group_idx = torch.repeat_interleave(torch.arange(ls.numel(), device=ls.device), ls)
    sums = torch.zeros(ls.numel(), dtype=flat.dtype, device=flat.device)
    sums.scatter_add_(0, group_idx, flat)
    y = sums / ls.to(dtype=flat.dtype)
    return y

def challenge_get_uniques(x: torch.Tensor) -> Tuple[Tensor, Tensor]:
    uniques, indices = (None, None)
    sorted_x, sorted_indices = torch.sort(x, stable=True)
    is_first = torch.ones_like(sorted_x, dtype=torch.bool)
    is_first[1:] = sorted_x[1:] != sorted_x[:-1]
    uniques = sorted_x[is_first]
    indices = sorted_indices[is_first]
    return (uniques, indices)
