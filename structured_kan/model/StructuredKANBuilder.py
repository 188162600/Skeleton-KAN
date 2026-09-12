"""Build explicit compositions from a small, independent tree specification."""
from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from torch import nn

from .GaussianBasis import GaussianBasis
from .KAN import KAN
from .StructuredKAN import StructuredKAN


class _Coordinate(nn.Module):
    def __init__(self, index, input_dim):
        super().__init__()
        self.index, self.input_dim = index, input_dim

    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} inputs, received {x.shape[-1]}")
        return x[..., self.index:self.index+1]


class _AffineRoute(nn.Module):
    def __init__(self, weights, bias, *, learnable, dtype, device):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=dtype, device=device).detach().clone()
        bias = weights.new_tensor(bias).reshape(())
        if weights.ndim != 1 or not bool(torch.isfinite(weights).all() & torch.isfinite(bias)):
            raise ValueError("affine route coefficients must be finite")
        for name, value in (("weights", weights), ("bias", bias)):
            if learnable:
                self.register_parameter(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)

    def forward(self, x):
        if x.shape[-1] != self.weights.numel():
            raise ValueError("affine route input dimension mismatch")
        return (torch.einsum("...i,i->...", x, self.weights) + self.bias).unsqueeze(-1)


class StructuredKANBuilder:
    """Build one model without symbolic parsing, clustering or fitting.

    The caller supplies each incoming phi's G. Repeated 'share' labels tie
    phi maps or affine routes; incompatible definitions raise an error.
    Registries reset for every build, so separate models are independent.
    Optional grid calibration uses only supplied training inputs. For a
    shared phi it pools all its incoming training activations.
    """

    def __init__(self, input_dim: int, *, default_G=3, dtype=torch.float64, device="cpu"):
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if type(default_G) is not int or default_G < 2:
            raise ValueError("default_G must be an integer >= 2")
        self.input_dim, self.default_G = input_dim, default_G
        self.dtype, self.device = dtype, torch.device(device)

    def input(self, index: int) -> nn.Module:
        if type(index) is not int or not 0 <= index < self.input_dim:
            raise ValueError("input index is outside the declared input set")
        return _Coordinate(index, self.input_dim)

    def affine_route(self, weights, bias=0.0, *, learnable=True) -> nn.Module:
        if len(weights) != self.input_dim:
            raise ValueError("affine route needs one weight per input")
        return _AffineRoute(weights, bias, learnable=learnable, dtype=self.dtype, device=self.device)

    def unary(self, G=None, *, low=-1.0, high=1.0, r=None, learn_centers=False, learn_width=False) -> KAN:
        basis = GaussianBasis.uniform(self.default_G if G is None else G, low, high, r=r,
            learn_centers=learn_centers, learn_width=learn_width, dtype=self.dtype, device=self.device)
        return KAN(basis)

    @staticmethod
    def node(children, phi, operation="sum", *, product_reduction="native") -> StructuredKAN:
        return StructuredKAN(children, phi, operation,product_reduction=product_reduction)

    def build_finite_topology(self,tree,*,G=None,per_map_G=None,output_G=None,train_inputs=None):
        """Build a finite baseline tree with explicit optional output phi.

        Shared-reserve catalogue rules are not interchangeable with a finite
        tree, and must not be passed to this method.
        """
        from .topology_spec import finite_topology_spec
        specification=finite_topology_spec(tree,self.input_dim,G=self.default_G if G is None else G,
                                           per_map_G=per_map_G,output_G=output_G)
        return self.build(specification,train_inputs=train_inputs)

    def build(self, specification: Mapping, *, train_inputs: torch.Tensor | None = None) -> StructuredKAN:
        """Materialize a JSON-compatible tree with explicit children and phi.

        Leaf: {'input': i} or {'affine': [weights], 'bias': 0, 'share': name}.
        Node: {'operation': 'sum'|'prod', 'children': [...], 'phi': [G|dict|'identity', ...]}.
        Map: {'G': 9, 'share': name, 'range': [lo, hi], 'r': ratio}.
        'range' fixes the grid interval; otherwise train_inputs calibrates it.
        Gaussian coefficients start at zero. Maps default to identity; optional
        initial_a/initial_b express source-frozen affine or neutral priors.
        """
        if not isinstance(specification, Mapping):
            raise TypeError("specification must be a mapping")
        samples = None
        if train_inputs is not None:
            samples = train_inputs.detach().to(dtype=self.dtype, device=self.device)
            if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != self.input_dim:
                raise ValueError("train_inputs must have shape (samples, input_dim)")
            if not bool(torch.isfinite(samples).all()):
                raise ValueError("train_inputs must be finite")
        routes, maps, calibration = {}, {}, {}

        def shared(registry, key, signature, create):
            if key is None:
                return create()
            if not isinstance(key, str) or not key:
                raise ValueError("share must be a nonempty string")
            if key in registry:
                previous, module = registry[key]
                if previous != signature:
                    raise ValueError(f"conflicting specifications for shared object {key!r}")
                return module
            module = create()
            registry[key] = (signature, module)
            return module

        def reject_extra(spec, allowed):
            extra = set(spec) - allowed
            if extra:
                raise ValueError(f"unknown specification keys: {sorted(extra)}")

        def phi(spec, child):
            if spec == "identity":
                return nn.Identity()
            if type(spec) is int:
                spec = {"G": spec}
            if not isinstance(spec, Mapping):
                raise TypeError("phi must be a G integer, map specification, or 'identity'")
            reject_extra(spec, {"G", "share", "range", "r", "learn_centers", "learn_width", "initial_a", "initial_b"})
            if "range" in spec and (not isinstance(spec["range"], (tuple, list)) or len(spec["range"]) != 2):
                raise ValueError("range must contain exactly two endpoints")
            config = dict(G=spec.get("G", self.default_G),
                low=spec.get("range", [-1.0, 1.0])[0], high=spec.get("range", [-1.0, 1.0])[1],
                r=spec.get("r"), learn_centers=spec.get("learn_centers", False),
                learn_width=spec.get("learn_width", False))
            signature = dict(config, adaptive="range" not in spec,
                             initial_a=spec.get('initial_a',1.0),initial_b=spec.get('initial_b',0.0))
            def create_edge():
                edge=self.unary(**config)
                with torch.no_grad():
                    edge.a.fill_(signature['initial_a']);edge.b.fill_(signature['initial_b'])
                if not bool(torch.isfinite(edge.a) & torch.isfinite(edge.b)):
                    raise ValueError('Initial affine phi coefficients must be finite')
                return edge
            edge = shared(maps, spec.get("share"), signature, create_edge)
            if samples is not None and "range" not in spec:
                with torch.no_grad():
                    values = child(samples).detach().reshape(-1)
                calibration.setdefault(id(edge), (edge, config["r"], []))[2].append(values)
            return edge

        def build(spec):
            if not isinstance(spec, Mapping):
                raise TypeError("every child specification must be a mapping")
            if "input" in spec:
                reject_extra(spec, {"input"})
                return self.input(spec["input"])
            if "affine" in spec:
                reject_extra(spec, {"affine", "bias", "learnable", "share"})
                signature = dict(weights=list(spec["affine"]), bias=spec.get("bias", 0.0),
                                 learnable=spec.get("learnable", True))
                return shared(routes, spec.get("share"), signature, lambda: self.affine_route(**signature))
            reject_extra(spec, {"operation", "children", "phi", "product_reduction"})
            if "children" not in spec or "phi" not in spec or "operation" not in spec:
                raise ValueError("each interaction node requires operation, children and phi")
            children = [build(child) for child in spec["children"]]
            if len(children) != len(spec["phi"]):
                raise ValueError("exactly one phi specification is required per child")
            edges = [phi(edge, child) for edge, child in zip(spec["phi"], children)]
            return self.node(children, edges, spec["operation"],product_reduction=spec.get('product_reduction','native'))

        model = build(copy.deepcopy(specification))
        if not isinstance(model, StructuredKAN):
            raise ValueError("the root must be an interaction node; use a unary node for one phi")
        # Gaussian coefficients start at zero, so calibrating their geometry
        # does not change parent activations, including neutral affine maps.
        for edge, ratio, values in calibration.values():
            edge.basis.initialize_from_samples(torch.cat(values), r=ratio)
        return model
