"""
Hash Center Generation Module

This module provides functionality for generating and managing hash centers
for deep hashing applications. It supports both Hadamard matrix-based and
random Bernoulli sampling methods for hash center generation.

Components:
- HashCenterGenerator: Orchestrates hash center generation, caching, and persistence
- HadamardMatrixBuilder: Constructs Hadamard matrices and extracts hash centers
- RandomBernoulliSampler: Generates random binary codes with distance constraints
- HashCenterValidator: Validates hash center quality through Hamming distance analysis
- CentroidCalculator: Computes hash centers for multi-label samples
- HashCenterDataset: PyTorch Dataset class with hash center assignment
"""

from .generator import HashCenterGenerator
from .builder import HadamardMatrixBuilder
from .sampler import RandomBernoulliSampler
from .validator import HashCenterValidator
from .calculator import CentroidCalculator
from .dataset import HashCenterDataset

__all__ = [
    'HashCenterGenerator',
    'HadamardMatrixBuilder',
    'RandomBernoulliSampler',
    'HashCenterValidator',
    'CentroidCalculator',
    'HashCenterDataset',
]
