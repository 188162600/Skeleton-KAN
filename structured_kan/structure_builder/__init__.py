from .catalogue import Catalogue


def build_catalogue(*args, **kwargs):
    """Load optional native-construction dependencies only when building."""
    from .build import build_catalogue as construct
    return construct(*args, **kwargs)

__all__ = ["Catalogue", "build_catalogue"]
