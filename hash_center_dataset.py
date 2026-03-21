"""
HashCenterDataset

PyTorch Dataset class that extends ImageList with hash center assignment functionality.
Maintains backward compatibility: when hash center parameters are not provided,
behaves identically to ImageList.
"""

import torch
import torch.utils.data as data
import numpy as np
import logging
from typing import Optional, Union, Tuple
from data_list import ImageList, make_dataset, default_loader
from hash_center import HashCenterGenerator, CentroidCalculator

logger = logging.getLogger(__name__)


class HashCenterDataset(data.Dataset):
    """
    PyTorch Dataset that extends ImageList with hash center assignment.
    
    Maintains backward compatibility: when hash center parameters are not
    provided, behaves identically to ImageList.
    """
    
    def __init__(self, 
                 image_list: list,
                 labels: Optional[np.ndarray] = None,
                 transform=None,
                 target_transform=None,
                 loader=default_loader,
                 # Hash center parameters
                 num_classes: Optional[int] = None,
                 hash_bit: Optional[int] = None,
                 hash_method: str = 'hadamard',
                 dataset_type: str = 'auto',
                 dataset_name: Optional[str] = None,
                 hash_centers_path: Optional[str] = None,
                 enable_hash_centers: bool = False,
                 validate_hash_centers: bool = True,
                 save_hash_centers: bool = False,
                 save_path: Optional[str] = None):
        """
        Initialize the dataset with optional hash center functionality.
        
        Args:
            image_list: List of image paths or list of strings with path and label
            labels: Optional numpy array of labels [num_samples, num_classes] for multi-label
            transform: Optional image transformation
            target_transform: Optional target transformation
            loader: Image loading function
            num_classes: Number of classes (required if enable_hash_centers=True)
            hash_bit: Hash code length (required if enable_hash_centers=True)
            hash_method: 'hadamard' or 'random'
            dataset_type: 'single-label', 'multi-label', or 'auto' (auto-detect)
            dataset_name: Optional dataset identifier for caching
            hash_centers_path: Path to load pre-generated hash centers
            enable_hash_centers: Whether to enable hash center functionality
            validate_hash_centers: Whether to validate generated hash centers
            save_hash_centers: Whether to save generated hash centers to disk
            save_path: Path to save hash centers (required if save_hash_centers=True)
        
        Raises:
            ValueError: If required parameters are missing when enable_hash_centers=True
        """
        # Initialize base dataset functionality using ImageList logic
        self.imgs = make_dataset(image_list, labels)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images in the provided image_list")
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        # Store hash center configuration
        self.enable_hash_centers = enable_hash_centers
        self.num_classes = num_classes
        self.hash_bit = hash_bit
        self.hash_method = hash_method
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.validate_hash_centers = validate_hash_centers
        self.save_hash_centers = save_hash_centers
        self.save_path = save_path
        self.hash_centers_path = hash_centers_path
        
        # Initialize hash centers if enabled
        self.class_hash_centers = None
        if self.enable_hash_centers:
            # Validate required parameters (Requirement 8.3)
            if self.num_classes is None or self.hash_bit is None:
                raise ValueError(
                    "num_classes and hash_bit are required when enable_hash_centers=True"
                )
            
            # Validate save_path if save_hash_centers is True
            if self.save_hash_centers and self.save_path is None:
                raise ValueError(
                    "save_path is required when save_hash_centers=True"
                )
            
            # Detect dataset type if set to 'auto'
            if self.dataset_type == 'auto':
                self.dataset_type = self._detect_dataset_type()
            
            # Initialize hash centers
            self._initialize_hash_centers()
            
            logger.info(
                f"HashCenterDataset initialized with {len(self.imgs)} samples, "
                f"{self.num_classes} classes, {self.hash_bit}-bit hash codes, "
                f"dataset_type={self.dataset_type}"
            )
    
    def __getitem__(self, index: int):
        """
        Get a sample from the dataset.
        
        Returns:
            If enable_hash_centers=False: (image, label)
            If enable_hash_centers=True and single-label: (image, label, hash_center)
            If enable_hash_centers=True and multi-label: (image, label, hash_center)
        
        Where:
            image: Transformed PIL image tensor
            label: Integer class index (single-label) or binary vector (multi-label)
            hash_center: Tensor of shape [hash_bit] with values in {-1, 1}
        """
        # Get image path and label
        path, target = self.imgs[index]
        
        # Load image
        img = self.loader(path)
        
        # Apply transformations
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Return based on hash center configuration
        if not self.enable_hash_centers:
            # Backward compatibility: return (image, label) (Requirement 8.5)
            return img, target
        else:
            # Get hash center for this sample (Requirement 8.6)
            hash_center = self._get_hash_center_for_sample(target)
            return img, target, hash_center
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.imgs)
    
    def _detect_dataset_type(self) -> str:
        """
        Auto-detect dataset type from label format.
        
        Returns:
            'single-label' if labels are integers
            'multi-label' if labels are arrays/vectors
        """
        # Sample the first label to determine type
        _, first_label = self.imgs[0]
        
        # Check if label is an integer (single-label) or array (multi-label)
        if isinstance(first_label, (int, np.integer)):
            detected_type = 'single-label'
        elif isinstance(first_label, (np.ndarray, torch.Tensor)):
            detected_type = 'multi-label'
        else:
            # Try to infer from type
            try:
                # If it's iterable and has length > 1, assume multi-label
                if hasattr(first_label, '__len__') and len(first_label) > 1:
                    detected_type = 'multi-label'
                else:
                    detected_type = 'single-label'
            except:
                # Default to single-label
                detected_type = 'single-label'
        
        logger.info(f"Auto-detected dataset type: {detected_type}")
        return detected_type
    
    def _initialize_hash_centers(self):
        """Initialize hash centers through generation or loading."""
        # Create hash center generator
        generator = HashCenterGenerator(
            num_classes=self.num_classes,
            hash_bit=self.hash_bit,
            method=self.hash_method
        )
        
        # Load from file if path provided (Requirement 10.5)
        if self.hash_centers_path is not None:
            logger.info(f"Loading hash centers from {self.hash_centers_path}")
            self.class_hash_centers = generator.load(self.hash_centers_path)
        else:
            # Generate new hash centers (Requirement 6.1)
            logger.info(f"Generating hash centers using {self.hash_method} method")
            self.class_hash_centers = generator.generate(
                dataset_name=self.dataset_name,
                validate=self.validate_hash_centers,
                use_cache=True
            )
            
            # Save to disk if requested (Requirement 6.5)
            if self.save_hash_centers:
                logger.info(f"Saving hash centers to {self.save_path}")
                metadata = {
                    'dataset_name': self.dataset_name,
                    'dataset_type': self.dataset_type,
                }
                generator.save(self.save_path, metadata=metadata)
    
    def _get_hash_center_for_sample(self, label: Union[int, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Get hash center for a sample based on its label.
        
        For single-label: directly index into class_hash_centers
        For multi-label: compute centroid using CentroidCalculator
        
        Args:
            label: Integer class index (single-label) or binary vector (multi-label)
        
        Returns:
            Tensor of shape [hash_bit] with values in {-1, 1}
        
        Raises:
            IndexError: If class index is invalid
        """
        if self.dataset_type == 'single-label':
            # Direct lookup for single-label (Requirement 3.2)
            class_idx = int(label)
            
            # Validate class index (Requirement 3.5)
            if class_idx < 0 or class_idx >= self.num_classes:
                raise IndexError(
                    f"Class index {class_idx} out of range for {self.num_classes} classes"
                )
            
            # Return hash center for this class (Requirement 3.4)
            return self.class_hash_centers[class_idx]
        
        else:  # multi-label
            # Convert label to tensor if needed
            if isinstance(label, np.ndarray):
                label_vector = torch.from_numpy(label).float()
            elif isinstance(label, torch.Tensor):
                label_vector = label.float()
            else:
                label_vector = torch.tensor(label, dtype=torch.float32)
            
            # Compute centroid for multi-label sample (Requirement 4.1, 4.2, 4.6)
            hash_center = CentroidCalculator.calculate(
                label_vector=label_vector,
                class_hash_centers=self.class_hash_centers
            )
            
            return hash_center
