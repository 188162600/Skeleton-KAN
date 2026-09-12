"""Fourier-MFN (ICLR 2021), matching official forward and initialization order.

Reference: boschresearch/multiplicative-filter-networks, revision
58c3c4dd5908bce9d5c3b077926cb588cc6052da. Scalar output; no output activation.
"""
import math
import torch
from torch import nn


def architecture_sweep():
    return [dict(id=f'MFN{i:02d}', product_layers=depth, hidden_width=width, input_scale=scale)
            for i, (depth, width, scale) in enumerate(
                ((d, w, s) for d in (1, 2, 4) for w in (32, 64, 128) for s in (1.0, 4.0)), 1)]


def parameter_count(input_dim, product_layers, hidden_width):
    return ((product_layers + 1) * (input_dim + 1) * hidden_width
            + product_layers * (hidden_width + 1) * hidden_width + hidden_width + 1)


class FourierFilter(nn.Module):
    def __init__(self, input_dim, width, scale, dtype):
        super().__init__()
        self.linear = nn.Linear(input_dim, width, dtype=dtype)
        with torch.no_grad():
            self.linear.weight.mul_(scale)
            self.linear.bias.uniform_(-math.pi, math.pi)

    def forward(self, inputs):
        return torch.sin(self.linear(inputs))


class FourierMFN(nn.Module):
    def __init__(self, input_dim, product_layers, hidden_width, input_scale,
                 *, seed=421, dtype=torch.float64, device='cpu'):
        super().__init__()
        if min(input_dim, product_layers, hidden_width) < 1 or input_scale <= 0:
            raise ValueError('dimensions, product depth and input scale must be positive')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.linear = nn.ModuleList([nn.Linear(hidden_width, hidden_width, dtype=dtype)
                                         for _ in range(product_layers)])
            self.output_linear = nn.Linear(hidden_width, 1, dtype=dtype)
            bound = math.sqrt(1.0 / hidden_width)
            with torch.no_grad():
                for layer in self.linear:
                    layer.weight.uniform_(-bound, bound)
            scale = input_scale / math.sqrt(product_layers + 1)
            self.filters = nn.ModuleList([FourierFilter(input_dim, hidden_width, scale, dtype)
                                          for _ in range(product_layers + 1)])
        self.to(device=device)
        assert sum(p.numel() for p in self.parameters()) == parameter_count(input_dim, product_layers, hidden_width)

    def forward(self, inputs):
        state = self.filters[0](inputs)
        for linear, filter_map in zip(self.linear, self.filters[1:]):
            state = filter_map(inputs) * linear(state)
        return self.output_linear(state)
