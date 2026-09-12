"""Standalone modelling and catalogue construction, with lazy Torch imports."""

__all__ = ["StructuredKAN", "KAN", "GaussianBasis", "StructuredKANBuilder", "MLP", "FourierMFN", "MultKAN"]


def __getattr__(name):
    if name in __all__:
        from . import model
        value = getattr(model, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
