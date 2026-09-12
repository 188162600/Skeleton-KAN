"""Gaussian basis geometry, independent of routes, maps and interaction nodes."""
from __future__ import annotations

import math

import torch
from torch import nn


def default_width_ratio(G: int) -> float:
    """Frozen sigma/grid-spacing ratio for one map's basis size."""
    if type(G) is not int or G < 2:
        raise ValueError("G must be an integer >= 2")
    return 0.2 * G ** (2.0 / 3.0)


class GaussianBasis(nn.Module):
    """Return G Gaussian activations for each scalar input.

    Centers and width are buffers by default, not trainable parameters.
    Coefficients and affine terms belong to KAN, not to this basis.
    """

    def __init__(self, centers, width, *, learn_centers=False, learn_width=False,
                 dtype=None, device=None):
        super().__init__()
        centers = torch.as_tensor(centers, dtype=dtype, device=device).detach().clone()
        if not centers.is_floating_point():
            centers = centers.to(torch.get_default_dtype())
        width = torch.as_tensor(width, dtype=centers.dtype, device=centers.device).detach().clone()
        if centers.ndim != 1 or centers.numel() < 2 or not bool(torch.isfinite(centers).all()):
            raise ValueError("centers must be a finite vector with at least two entries")
        if width.numel() != 1 or not bool(torch.isfinite(width).all()) or float(width) <= 0:
            raise ValueError("width must be a positive finite scalar")
        for name, tensor, learn in (("centers", centers, learn_centers),
                                    ("width", width.reshape(()), learn_width)):
            if learn:
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

    @property
    def G(self) -> int:
        return self.centers.numel()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        width = self.width.abs().clamp_min(1.0e-9)
        return torch.exp(-0.5 * ((z.unsqueeze(-1) - self.centers) / width).square())

    @classmethod
    def uniform(cls, G: int, low=-1.0, high=1.0, *, r=None, **options):
        ratio = default_width_ratio(G) if r is None else float(r)
        if type(G) is not int or G < 2:
            raise ValueError("G must be an integer >= 2")
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError("the grid interval must be finite and increasing")
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("r must be positive and finite")
        centers = torch.linspace(low, high, G, dtype=options.get("dtype"), device=options.get("device"))
        return cls(centers, max(ratio * (high-low) / (G-1), 1.0e-3), **options)

    @torch.no_grad()
    def initialize_from_samples(self, values: torch.Tensor, *, r=None, quantiles=(0.01, 0.99)):
        """Set geometry once from this map's incoming training activations.

        Explicit initialization only: forward() never updates the grid.
        Calling this does not change which tensors are trainable.
        """
        q0, q1 = quantiles
        if not 0 <= q0 < q1 <= 1:
            raise ValueError("quantiles must satisfy 0 <= q0 < q1 <= 1")
        values = values.detach().reshape(-1).to(device=self.centers.device, dtype=self.centers.dtype)
        if not values.numel() or not bool(torch.isfinite(values).all()):
            raise ValueError("grid initialization needs nonempty finite training activations")
        low, high = torch.quantile(values, values.new_tensor([q0, q1])).tolist()
        if low == high:
            low, high = low-1.0, high+1.0
        initialized = self.uniform(self.G, low, high, r=r,
                                   dtype=self.centers.dtype, device=self.centers.device)
        self.centers.copy_(initialized.centers)
        self.width.copy_(initialized.width)
        return self
