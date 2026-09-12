"""Published PyKAN MultKAN, using the verified numerical fast path.

The spline basis remains cubic; this is not the project's Gaussian KAN.
Disabled symbolic parameters and identity node affines are frozen explicitly.
"""
import copy
import random

import numpy as np
import torch
from torch import nn


def architecture_sweep():
    return [dict(id=f'MKAN{i:02d}', first_hidden=first, second_hidden=second,
                 grid=grid, spline_order=3, multiplication_arity=2)
            for i, (grid, first, second) in enumerate(
                ((g, h1, h2) for g in (10, 30)
                 for h1 in ('q_add', 'q_add_1_mult', 'q_plus_1_add_1_mult')
                 for h2 in ('none', '2_add', '1_mult')), 1)]


def widths(input_dim, first_hidden, second_hidden):
    if not isinstance(input_dim, int) or input_dim < 1:
        raise ValueError('input_dim must be a positive integer')
    q = (input_dim + 1) // 2
    first = {'q_add': [q, 0], 'q_add_1_mult': [q, 1],
             'q_plus_1_add_1_mult': [q+1, 1]}[first_hidden]
    second = {'none': [], '2_add': [[2, 0]], '1_mult': [[0, 1]]}[second_hidden]
    return [[input_dim, 0], first, *second, [1, 0]]


def parameter_count(input_dim, first_hidden, second_hidden, grid):
    width = widths(input_dim, first_hidden, second_hidden)
    edges = sum(sum(left) * (right[0] + 2*right[1])
                for left, right in zip(width, width[1:]))
    # G+3 cubic spline coefficients, plus SiLU and spline scales per edge.
    return edges * (grid + 5)


def native_model(input_dim, first_hidden, second_hidden, grid, *, seed=421,
                 optimized=True):
    if grid not in (10, 30):
        raise ValueError('this frozen sweep uses G=10 or G=30')
    if optimized:
        from ..vendor.pykan_optimized.kan.MultKAN import MultKAN as Native
    else:
        from ..vendor.pykan_upstream.kan.MultKAN import MultKAN as Native
    old_dtype = torch.get_default_dtype()
    old_random, old_numpy = random.getstate(), np.random.get_state()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=[]):
            model = Native(width=copy.deepcopy(widths(input_dim, first_hidden, second_hidden)),
                grid=grid, k=3, mult_arity=2, base_fun='silu', symbolic_enabled=False,
                affine_trainable=False, grid_eps=0.02, grid_range=[-1.0, 1.0],
                seed=seed, save_act=False, sparse_init=False, auto_save=False, device='cpu')
        for p in model.symbolic_fun.parameters():
            p.requires_grad_(False)
    finally:
        torch.set_default_dtype(old_dtype)
        random.setstate(old_random)
        np.random.set_state(old_numpy)
    return model


class MultKAN(nn.Module):
    def __init__(self, input_dim, first_hidden, second_hidden, grid, *,
                 seed=421, device='cpu'):
        super().__init__()
        self.network = native_model(input_dim, first_hidden, second_hidden, grid, seed=seed)
        self.network.to(str(device))
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) == parameter_count(
            input_dim, first_hidden, second_hidden, grid)

    def forward(self, inputs):
        return self.network(inputs)
