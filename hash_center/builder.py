"""
Hadamard Matrix Builder

Constructs Hadamard matrices and extracts hash centers.
"""

import torch
import numpy as np
from scipy.linalg import hadamard


class HadamardMatrixBuilder:
    """
    Builds hash centers using Hadamard matrices.
    
    The Hadamard matrix is an orthogonal matrix with entries in {-1, 1}.
    Each row provides maximally separated binary codes.
    """
    
    @staticmethod
    def build(num_classes: int, hash_bit: int) -> torch.Tensor:
        """
        Build hash centers from Hadamard matrix.
        
        Algorithm:
        1. Verify hash_bit is a power of 2
        2. Construct Hadamard matrix of dimension hash_bit
        3. If num_classes <= hash_bit: extract first num_classes rows
        4. If num_classes > hash_bit: concatenate matrix with its negation,
           then extract first num_classes rows
        
        Args:
            num_classes: Number of hash centers to generate
            hash_bit: Dimension of Hadamard matrix
        
        Returns:
            Tensor of shape [num_classes, hash_bit] with values in {-1, 1}
        
        Raises:
            ValueError: If hash_bit is not a power of 2
        """
        # Step 1: Verify hash_bit is a power of 2
        if not HadamardMatrixBuilder._is_power_of_two(hash_bit):
            raise ValueError(f"Hash bit length must be a power of 2 for Hadamard method. Got: {hash_bit}")
        
        # Step 2: Construct Hadamard matrix
        H = hadamard(hash_bit)
        
        # Step 3: Determine extraction strategy
        if num_classes <= hash_bit:
            # Extract first num_classes rows
            hash_centers = H[0:num_classes, :]
        else:
            # Concatenate matrix with its negation
            H_extended = np.concatenate([H, -H], axis=0)
            
            # Check if we have enough rows
            if num_classes > 2 * hash_bit:
                raise ValueError(
                    f"Cannot generate {num_classes} hash centers with hash_bit={hash_bit}. "
                    f"Maximum supported: {2 * hash_bit}"
                )
            
            # Extract first num_classes rows
            hash_centers = H_extended[0:num_classes, :]
        
        # Step 4: Convert to PyTorch tensor
        hash_centers = torch.from_numpy(hash_centers).float()
        
        return hash_centers
    
    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        """Check if n is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0
