"""
Hash Center Generator

Manages hash center generation, caching, and persistence.
"""

import torch
import pickle
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from .builder import HadamardMatrixBuilder
from .sampler import RandomBernoulliSampler
from .validator import HashCenterValidator

logger = logging.getLogger(__name__)


class HashCenterGenerator:
    """
    Manages hash center generation, caching, and persistence.
    
    Attributes:
        num_classes (int): Number of classes in the dataset
        hash_bit (int): Length of binary hash codes
        method (str): Generation method ('hadamard' or 'random')
        cache (dict): In-memory cache of generated hash centers
    """
    
    # Class-level cache shared across all instances
    _cache: Dict[str, torch.Tensor] = {}
    
    def __init__(self, num_classes: int, hash_bit: int, method: str = 'hadamard'):
        """
        Initialize the hash center generator.
        
        Args:
            num_classes: Number of classes in the dataset
            hash_bit: Length of binary hash codes (must be power of 2 for Hadamard)
            method: Generation method ('hadamard' or 'random')
        
        Raises:
            ValueError: If method is invalid or hash_bit is not power of 2 for Hadamard
        """
        # Validate method parameter (Requirement 5.4)
        if method not in ['hadamard', 'random']:
            raise ValueError(f"Invalid generation method: {method}. Must be 'hadamard' or 'random'")
        
        # Validate hash_bit for Hadamard method (Requirement 1.2)
        if method == 'hadamard' and not self._is_power_of_two(hash_bit):
            raise ValueError(f"Hash bit length must be a power of 2 for Hadamard method. Got: {hash_bit}")
        
        self.num_classes = num_classes
        self.hash_bit = hash_bit
        self.method = method
        self.hash_centers = None  # Store generated hash centers
    
    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        """Check if n is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0
    
    def generate(self, dataset_name: Optional[str] = None, 
                 validate: bool = True,
                 use_cache: bool = True) -> torch.Tensor:
        """
        Generate or retrieve hash centers.
        
        Args:
            dataset_name: Optional dataset identifier for cache key
            validate: Whether to validate hash center quality
            use_cache: Whether to use cached hash centers if available
        
        Returns:
            Tensor of shape [num_classes, hash_bit] with values in {-1, 1}
        """
        # Generate cache key (Requirement 6.3)
        cache_key = self._get_cache_key(dataset_name)
        
        # Check cache if enabled (Requirement 6.1, 6.2)
        if use_cache and cache_key in HashCenterGenerator._cache:
            logger.info(f"Loading hash centers from cache: {cache_key}")
            self.hash_centers = HashCenterGenerator._cache[cache_key]
            return self.hash_centers
        
        # Generate new hash centers based on method (Requirement 5.1, 5.2, 5.3)
        logger.info(f"Generating hash centers using {self.method} method...")
        
        if self.method == 'hadamard':
            # Use Hadamard matrix builder (Requirement 5.2)
            self.hash_centers = HadamardMatrixBuilder.build(self.num_classes, self.hash_bit)
        elif self.method == 'random':
            # Use random Bernoulli sampler (Requirement 5.3)
            sampler = RandomBernoulliSampler()
            self.hash_centers = sampler.sample(self.num_classes, self.hash_bit)
        
        # Validate hash centers if requested (Requirement 7.1-7.6)
        if validate:
            validation_stats = HashCenterValidator.validate(self.hash_centers, log_stats=True)
            logger.info(f"Hash centers generated successfully. Validation: {validation_stats['is_valid']}")
        
        # Store in cache (Requirement 6.1)
        if use_cache:
            HashCenterGenerator._cache[cache_key] = self.hash_centers
            logger.info(f"Hash centers cached with key: {cache_key}")
        
        return self.hash_centers
    
    def save(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save hash centers to disk with metadata.
        
        Args:
            filepath: Path to save pickle file
            metadata: Optional metadata dictionary
        """
        if self.hash_centers is None:
            raise ValueError("No hash centers to save. Call generate() first.")
        
        # Prepare metadata (Requirement 10.6)
        save_data = {
            'hash_centers': self.hash_centers,
            'num_classes': self.num_classes,
            'hash_bit': self.hash_bit,
            'method': self.method,
            'creation_timestamp': datetime.now().isoformat(),
        }
        
        # Add user-provided metadata if available
        if metadata:
            save_data['metadata'] = metadata
        
        # Save to pickle file (Requirement 6.5, 10.1)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Hash centers saved to {filepath}")
    
    def load(self, filepath: str) -> torch.Tensor:
        """
        Load hash centers from disk.
        
        Args:
            filepath: Path to pickle file
        
        Returns:
            Loaded hash centers tensor
        
        Raises:
            ValueError: If loaded hash centers have invalid shape or values
            FileNotFoundError: If file does not exist
        """
        # Load from pickle file (Requirement 6.5, 10.2)
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Hash centers file not found: {filepath}")
        
        # Extract hash centers
        hash_centers = save_data['hash_centers']
        
        # Validate shape (Requirement 10.3)
        expected_shape = (self.num_classes, self.hash_bit)
        if hash_centers.shape != expected_shape:
            raise ValueError(
                f"Loaded hash centers have shape {hash_centers.shape}, "
                f"expected {expected_shape}"
            )
        
        # Validate values are in {-1, 1} (Requirement 10.4)
        unique_values = torch.unique(hash_centers)
        valid_values = torch.tensor([-1.0, 1.0])
        if not all(val in valid_values for val in unique_values):
            raise ValueError(
                f"Hash centers must contain only values in {{-1, 1}}, "
                f"found: {unique_values.tolist()}"
            )
        
        self.hash_centers = hash_centers
        logger.info(f"Hash centers loaded from {filepath}")
        
        # Log metadata if available (Requirement 10.6)
        if 'creation_timestamp' in save_data:
            logger.info(f"  Created: {save_data['creation_timestamp']}")
        if 'method' in save_data:
            logger.info(f"  Method: {save_data['method']}")
        
        return hash_centers
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        HashCenterGenerator._cache.clear()
        logger.info("Hash center cache cleared")
    
    def _get_cache_key(self, dataset_name: Optional[str] = None) -> str:
        """
        Generate cache key from configuration.
        
        Algorithm from design document:
        1. Normalize dataset_name (default to "default" if None)
        2. Construct cache key: "{dataset_name}_{num_classes}_{hash_bit}_{method}"
        
        Args:
            dataset_name: Optional dataset identifier
        
        Returns:
            Cache key string (Requirement 6.3)
        """
        # Normalize dataset_name
        if dataset_name is None:
            dataset_name = "default"
        dataset_name = dataset_name.lower().replace(" ", "_")
        
        # Construct cache key
        cache_key = f"{dataset_name}_{self.num_classes}_{self.hash_bit}_{self.method}"
        
        return cache_key
