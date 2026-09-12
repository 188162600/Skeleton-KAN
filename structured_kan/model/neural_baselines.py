"""Shared construction/replay interface for the ordinary neural baselines."""
from .MLP import MLP, parameter_hash
from .FourierMFN import FourierMFN


def architecture_sweep(method):
    if method == 'mlp':
        from .MLP import architecture_sweep as sweep
    elif method == 'fourier_mfn':
        from .FourierMFN import architecture_sweep as sweep
    elif method == 'multkan':
        from .MultKAN import architecture_sweep as sweep
    elif method == 'smpf_structure':
        from ..structure_builder.smpf import architecture_sweep as sweep
    else:
        raise ValueError(method)
    return sweep()


def build_model(method, input_dim, architecture, *, seed=421, device='cpu', train_inputs=None):
    if method == 'smpf_structure':
        from .SMPFStructure import build_model as build
        return build(input_dim, architecture, train_inputs=train_inputs, seed=seed, device=device)
    if method == 'mlp':
        keys, cls = ('hidden_layers', 'hidden_width', 'activation'), MLP
    elif method == 'fourier_mfn':
        keys, cls = ('product_layers', 'hidden_width', 'input_scale'), FourierMFN
    elif method == 'multkan':
        from .MultKAN import MultKAN
        keys, cls = ('first_hidden', 'second_hidden', 'grid'), MultKAN
    else:
        raise ValueError(method)
    return cls(input_dim, **{key: architecture[key] for key in keys}, seed=seed, device=device)


def expected_parameters(method, input_dim, architecture):
    if method == 'smpf_structure':
        from ..structure_builder.smpf import parameter_count
        return parameter_count(input_dim, architecture)
    if method == 'mlp':
        from .MLP import parameter_count
        depth = architecture['hidden_layers']
    elif method == 'fourier_mfn':
        from .FourierMFN import parameter_count
        depth = architecture['product_layers']
    elif method == 'multkan':
        from .MultKAN import parameter_count
        return parameter_count(input_dim, architecture['first_hidden'],
                               architecture['second_hidden'], architecture['grid'])
    else:
        raise ValueError(method)
    return parameter_count(input_dim, depth, architecture['hidden_width'])
