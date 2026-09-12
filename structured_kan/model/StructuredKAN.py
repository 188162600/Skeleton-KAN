"""An interaction node owns children AND the phi maps on its incoming edges."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn


class StructuredKAN(nn.Module):
    """sum_i phi[i](children[i](x)) or prod_i phi[i](children[i](x)).

    Children may be input routes or other StructuredKAN nodes. Each incoming
    edge has an explicit map, including nn.Identity for an unmapped edge.
    No phi is applied after the reduction and no final affine is inserted.
    Reusing a child/map object shares its parameters; copying it does not.
    """

    def __init__(self, children: Iterable[nn.Module], phi: Iterable[nn.Module], operation="sum", *, product_reduction="native"):
        super().__init__()
        if operation not in {"sum", "prod"}:
            raise ValueError("operation must be 'sum' or 'prod'")
        for name, values in (("children", children), ("phi", phi)):
            if isinstance(values, (nn.Module, Mapping, str, bytes)):
                raise TypeError(f"{name} must be an iterable of modules")
        children, phi = list(children), list(phi)
        if not children or not all(isinstance(child, nn.Module) for child in children):
            raise ValueError("at least one child module is required")
        if len(phi) != len(children) or not all(isinstance(edge, nn.Module) for edge in phi):
            raise ValueError("exactly one phi module is required per child")
        # Preserve nn.Module.children(), which PyTorch uses for traversal.
        self.branches = nn.ModuleList(children)
        self.phi = nn.ModuleList(phi)
        self.operation = operation
        if product_reduction not in ('native','sequential'):
            raise ValueError('product_reduction must be native or sequential')
        self.product_reduction=product_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values=[edge(child(x)) for child, edge in zip(self.branches, self.phi)]
        if self.operation=='prod' and self.product_reduction=='sequential':
            # Polynomial product rule at all orders, including zero factors;
            # avoids division/count-zero branches in torch.prod's derivatives.
            result=values[0]
            for value in values[1:]:result=result*value
            return result
        mapped=torch.stack(values,dim=0)
        return mapped.sum(0) if self.operation == "sum" else mapped.prod(0)

    def parameter_count(self, *, trainable_only=True) -> int:
        """Count unique tensors, not repeated occurrences of shared modules."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def extra_repr(self):
        return f"operation={self.operation!r}, edges={len(self.phi)}"
