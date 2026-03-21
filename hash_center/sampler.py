"""
Random Bernoulli Sampler

Generates random binary codes with distance constraints.
"""

import torch


class RandomBernoulliSampler:
    """
    Generates hash centers through random Bernoulli sampling.
    
    Ensures generated codes have:
    - Balanced bits (half 1s, half -1s)
    - Minimum Hamming distance >= 20 bits
    - Average Hamming distance >= hash_bit / 2
    """
    
    def __init__(self, min_hamming_distance: int = 20,
                 max_attempts: int = 10000):
        """
        Initialize the random sampler.
        
        Args:
            min_hamming_distance: Minimum Hamming distance between any two codes
            max_attempts: Maximum regeneration attempts before failure
        """
        self.min_hamming_distance = min_hamming_distance
        self.max_attempts = max_attempts
    
    def sample(self, num_classes: int, hash_bit: int) -> torch.Tensor:
        """
        Sample random hash centers with distance constraints.
        
        Algorithm:
        1. For each class, randomly select hash_bit/2 positions to set to -1
        2. Set remaining positions to 1
        3. Compute pairwise Hamming distances
        4. If constraints not met, regenerate
        5. Repeat until constraints satisfied or max_attempts reached
        
        Args:
            num_classes: Number of hash centers to generate
            hash_bit: Length of binary hash codes
        
        Returns:
            Tensor of shape [num_classes, hash_bit] with values in {-1, 1}
        
        Raises:
            RuntimeError: If unable to generate valid hash centers within max_attempts
        """
        for attempt in range(self.max_attempts):
            hash_centers = torch.zeros(num_classes, hash_bit)
            for i in range(num_classes):
                hash_centers[i] = self._generate_single_code(hash_bit)
            
            if self._check_constraints(hash_centers, hash_bit):
                return hash_centers
        
        raise RuntimeError(
            f"Failed to generate hash centers meeting distance constraints after "
            f"{self.max_attempts} attempts. Try increasing hash_bit or reducing num_classes."
        )
    
    def _generate_single_code(self, hash_bit: int) -> torch.Tensor:
        """Generate a single balanced binary code."""
        code = torch.ones(hash_bit)
        positions = torch.randperm(hash_bit)[:hash_bit // 2]
        code[positions] = -1
        return code
    
    def _check_constraints(self, hash_centers: torch.Tensor, 
                          hash_bit: int) -> bool:
        """Check if hash centers meet distance constraints."""
        num_classes = hash_centers.shape[0]
        if num_classes < 2:
            return True
        
        distances = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                dist = torch.sum(hash_centers[i] != hash_centers[j]).item()
                distances.append(dist)
        
        min_dist = min(distances)
        avg_dist = sum(distances) / len(distances)
        
        return min_dist >= self.min_hamming_distance and avg_dist >= hash_bit / 2
