"""The SMPF-inspired structure ablation uses the existing model primitives."""
from .StructuredKANBuilder import StructuredKANBuilder
from .initialization import perturb_trainable_
from ..structure_builder.smpf import specification, parameter_count


def build_model(input_dim, architecture, *, train_inputs, seed=421, device='cpu'):
    if train_inputs is None:
        raise ValueError('Training inputs are required for train-only grid calibration')
    spec = specification(input_dim, architecture)
    model = StructuredKANBuilder(input_dim, device=device).build(spec, train_inputs=train_inputs)
    model.initialization_sha256 = perturb_trainable_(model, seed, .001)
    model.structure_specification = spec
    assert model.parameter_count() == parameter_count(input_dim, architecture)
    return model
