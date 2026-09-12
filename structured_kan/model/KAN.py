"""A scalar unary map: affine base plus weighted Gaussian activations."""
from __future__ import annotations

import torch
from torch import nn

from .GaussianBasis import GaussianBasis


class KAN(nn.Module):
    """phi(z) = a*z + b + sum_g c[g]*GaussianBasis(z)[g].

    Owns G+2 trainable scalars with a frozen basis. There is no hidden input
    projection, extra input affine, or additional output affine. Input routes
    and the parent sum/product belong to StructuredKAN's construction.
    """

    def __init__(self, basis: GaussianBasis, *, coefficients=None, a=1.0, b=0.0):
        super().__init__()
        if not isinstance(basis, GaussianBasis):
            raise TypeError("basis must be a GaussianBasis")
        self.basis = basis
        coefficient = torch.zeros_like(basis.centers) if coefficients is None else torch.as_tensor(
            coefficients, dtype=basis.centers.dtype, device=basis.centers.device)
        if coefficient.shape != basis.centers.shape or not bool(torch.isfinite(coefficient).all()):
            raise ValueError("one finite coefficient is required per Gaussian center")
        self.coefficients = nn.Parameter(coefficient.detach().clone())
        self.a = nn.Parameter(coefficient.new_tensor(a).reshape(()))
        self.b = nn.Parameter(coefficient.new_tensor(b).reshape(()))
        if not bool(torch.isfinite(self.a) & torch.isfinite(self.b)):
            raise ValueError("affine coefficients must be finite")

    @property
    def G(self) -> int:
        return self.basis.G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.a*z + self.b + torch.einsum("...g,g->...", self.basis(z), self.coefficients)
