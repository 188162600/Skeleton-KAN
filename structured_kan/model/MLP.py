"""Ordinary dense MLP baseline; seeded Xavier weights and zero biases."""
import hashlib

import torch
from torch import nn


def architecture_sweep():
    return [dict(id=f'MLP{i:02d}', hidden_layers=depth, hidden_width=width,
                 activation=activation)
            for i, (depth, width, activation) in enumerate(
                ((d, w, a) for d in (1, 2, 4) for w in (32, 64, 128)
                 for a in ('tanh', 'silu')), 1)]


def parameter_count(input_dim, hidden_layers, hidden_width):
    return ((input_dim + 1) * hidden_width
            + (hidden_layers - 1) * (hidden_width + 1) * hidden_width
            + hidden_width + 1)


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_width, activation,
                 *, seed=421, dtype=torch.float64, device='cpu'):
        super().__init__()
        if min(input_dim, hidden_layers, hidden_width) < 1:
            raise ValueError('dimensions and hidden depth must be positive')
        if activation not in ('tanh', 'silu', 'relu'):
            raise ValueError('activation must be tanh, silu or relu')
        # CPU initialization gives reproducible starts independently of GPU.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers = []
            width = input_dim
            for _ in range(hidden_layers):
                layers.extend([nn.Linear(width, hidden_width, dtype=dtype),
                               {'tanh': nn.Tanh, 'silu': nn.SiLU, 'relu': nn.ReLU}[activation]()])
                width = hidden_width
            layers.append(nn.Linear(width, 1, dtype=dtype))
            self.network = nn.Sequential(*layers)
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        self.to(device=device)
        assert sum(p.numel() for p in self.parameters()) == parameter_count(
            input_dim, hidden_layers, hidden_width)

    def forward(self, inputs):
        return self.network(inputs)
