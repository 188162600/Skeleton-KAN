"""Standalone compositions of parameterized unary maps. Requires PyTorch only."""
from .GaussianBasis import GaussianBasis
from .KAN import KAN
from .StructuredKAN import StructuredKAN
from .StructuredKANBuilder import StructuredKANBuilder
from .MLP import MLP
from .FourierMFN import FourierMFN
from .MultKAN import MultKAN

__all__ = ["StructuredKAN", "KAN", "GaussianBasis", "StructuredKANBuilder", "MLP", "FourierMFN", "MultKAN"]
