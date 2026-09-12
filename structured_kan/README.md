# Structured KAN review package

See the [release README](../README.md) for installation, reconstruction, training and reporting commands. `data/train_config/` contains the paper configurations; `data/catalogues/` contains one JSON per frozen skeleton group. Training creates `data/results/` when needed.

The model exposes `StructuredKAN(children=..., phi=..., operation="sum" or "prod")`: it applies the incoming unary maps to its children, then reduces them. There is no implicit post-reduction map. Explicit output maps and parameter sharing are specified by the builder. Each frozen-grid unary map has `G+2` trainable parameters; each private affine route has `d+1` parameters. Shared tensors are counted once.

The mathematical projection and numerical kernel realization are separate components. Source-side metadata supplies optional complexity annotations; the training configuration selects the per-map component allocation. No held-out formula enters source-side construction.
