"""
Centroid Calculator

Computes hash centers for multi-label samples.
"""

import torch


class CentroidCalculator:
    """
    Calculates hash centers for multi-label samples.
    
    For samples with multiple active labels, computes the centroid
    of corresponding class hash centers and binarizes the result.
    """
    
    @staticmethod
    def calculate(label_vector: torch.Tensor, 
                  class_hash_centers: torch.Tensor) -> torch.Tensor:
        """
        Calculate hash center for a multi-label sample.
        
        Algorithm:
        1. Extract hash centers for all active classes (label_vector == 1)
        2. Compute element-wise mean across selected hash centers
        3. Binarize: set positive values to 1, negative values to -1
        4. For zero values, randomly assign -1 or 1
        
        Args:
            label_vector: Binary vector of shape [num_classes] indicating active labels
            class_hash_centers: Tensor of shape [num_classes, hash_bit]
        
        Returns:
            Tensor of shape [hash_bit] representing the sample's hash center
        
        Raises:
            ValueError: If no active labels in label_vector
        """
        # Extract active class indices
        active_indices = (label_vector == 1).nonzero(as_tuple=True)[0]
        
        if len(active_indices) == 0:
            raise ValueError("Multi-label sample must have at least one active label")
        
        # Extract hash centers for active classes
        active_centers = class_hash_centers[active_indices]
        
        # Compute element-wise mean
        centroid = active_centers.mean(dim=0)
        
        # Binarize the mean to produce final hash center
        sample_hash_center = CentroidCalculator._binarize(centroid)
        
        return sample_hash_center
    
    @staticmethod
    def _binarize(values: torch.Tensor) -> torch.Tensor:
        """
        Binarize continuous values to {-1, 1}.
        
        Positive values -> 1
        Negative values -> -1
        Zero values -> random choice from {-1, 1}
        """
        result = torch.zeros_like(values)
        result[values > 0] = 1
        result[values < 0] = -1
        
        # Handle zero values with random assignment
        zero_mask = (values == 0)
        if zero_mask.any():
            num_zeros = zero_mask.sum().item()
            random_values = torch.randint(0, 2, (num_zeros,), dtype=values.dtype, device=values.device)
            random_values = random_values * 2 - 1  # Convert 0,1 to -1,1
            result[zero_mask] = random_values
        
        return result
