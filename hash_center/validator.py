"""
Hash Center Validator

Validates hash center quality through Hamming distance analysis.
"""

import torch
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HashCenterValidator:
    """
    Validates hash center quality through Hamming distance analysis.
    """
    
    @staticmethod
    def validate(hash_centers: torch.Tensor, 
                 log_stats: bool = True) -> Dict:
        """
        Validate hash centers and compute distance statistics.
        
        Args:
            hash_centers: Tensor of shape [num_classes, hash_bit]
            log_stats: Whether to log statistics
        
        Returns:
            Dictionary containing:
            - min_distance: Minimum Hamming distance
            - max_distance: Maximum Hamming distance
            - avg_distance: Average Hamming distance
            - std_distance: Standard deviation of distances
            - is_valid: Boolean indicating if centers meet quality thresholds
        """
        # Compute pairwise Hamming distances
        distances = HashCenterValidator.compute_pairwise_hamming_distances(hash_centers)
        
        # Compute statistics
        min_distance = float(np.min(distances))
        max_distance = float(np.max(distances))
        avg_distance = float(np.mean(distances))
        std_distance = float(np.std(distances))
        
        # Determine if hash centers meet quality thresholds
        # Good separation: min_distance should be reasonably high
        hash_bit = hash_centers.shape[1]
        is_valid = min_distance >= max(10, hash_bit * 0.15)  # At least 15% of hash_bit or 10 bits
        
        # Log statistics if requested (Requirement 7.5)
        if log_stats:
            logger.info(f"Hash Center Validation Statistics:")
            logger.info(f"  Min Hamming Distance: {min_distance}")
            logger.info(f"  Max Hamming Distance: {max_distance}")
            logger.info(f"  Avg Hamming Distance: {avg_distance:.2f}")
            logger.info(f"  Std Hamming Distance: {std_distance:.2f}")
            logger.info(f"  Hash Bit Length: {hash_bit}")
            logger.info(f"  Number of Classes: {hash_centers.shape[0]}")
        
        # Issue warning for poor separation (Requirement 7.6)
        if not is_valid:
            logger.warning(
                f"Poor hash center separation detected! "
                f"Minimum Hamming distance ({min_distance}) is below recommended threshold "
                f"({max(10, hash_bit * 0.15):.1f}). Consider increasing hash_bit or reducing num_classes."
            )
        
        return {
            'min_distance': min_distance,
            'max_distance': max_distance,
            'avg_distance': avg_distance,
            'std_distance': std_distance,
            'is_valid': is_valid
        }
    
    @staticmethod
    def compute_pairwise_hamming_distances(hash_centers: torch.Tensor) -> np.ndarray:
        """
        Compute pairwise Hamming distances between all hash centers.
        
        Algorithm from design document:
        1. Initialize distances array with size num_pairs = num_classes * (num_classes - 1) / 2
        2. Compute pairwise distances using nested loops
        3. Return distances array
        
        Args:
            hash_centers: Tensor of shape [num_classes, hash_bit]
        
        Returns:
            Array of shape [num_pairs] containing Hamming distances
        """
        num_classes = hash_centers.shape[0]
        num_pairs = num_classes * (num_classes - 1) // 2
        distances = np.zeros(num_pairs, dtype=np.float64)
        
        idx = 0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                distances[idx] = HashCenterValidator._hamming_distance(
                    hash_centers[i], hash_centers[j]
                )
                idx += 1
        
        return distances
    
    @staticmethod
    def _hamming_distance(code1: torch.Tensor, code2: torch.Tensor) -> int:
        """
        Compute Hamming distance between two binary codes.
        
        Hamming distance is the number of positions at which corresponding bits differ.
        
        Args:
            code1: First binary code tensor
            code2: Second binary code tensor
        
        Returns:
            Hamming distance as an integer
        """
        return int(torch.sum(code1 != code2).item())
