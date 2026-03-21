"""
HashCenterDataset

PyTorch Dataset class that extends ImageList with hash center assignment.
"""

import torch
import numpy as np
import logging
from typing import Optional, Callable, Tuple, Union
from PIL import Image
from .generator import HashCenterGenerator
from .calculator import CentroidCalculator

logger = logging.getLogger(__name__)


def _pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# Local copy — avoids circular / relative import from data_list
def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


# Re-export for backward compat
default_loader = _pil_loader


class HashCenterDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset that extends ImageList with hash center assignment.
    
    Maintains backward compatibility: when hash center parameters are not
    provided, behaves identically to ImageList.
    """
    
    def __init__(self, 
                 image_list: list,
                 labels: Optional[np.ndarray] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable = default_loader,
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
        # Validate required parameters when hash centers are enabled (Requirement 8.3)
        if enable_hash_centers:
            if num_classes is None or hash_bit is None:
                raise ValueError(
                    "num_classes and hash_bit are required when enable_hash_centers=True"
                )
            if save_hash_centers and save_path is None:
                raise ValueError(
                    "save_path is required when save_hash_centers=True"
                )
        
        # Initialize ImageList components (Requirement 8.1, 8.2)
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
        self.dataset_name = dataset_name
        self.validate_hash_centers = validate_hash_centers
        self.save_hash_centers = save_hash_centers
        self.save_path = save_path
        self.hash_centers_path = hash_centers_path
        
        # Initialize hash centers if enabled
        self.class_hash_centers = None
        self.dataset_type = None
        
        if self.enable_hash_centers:
            # Detect dataset type (Requirement 9.1, 9.2, 9.3, 9.4)
            if dataset_type == 'auto':
                self.dataset_type = self._detect_dataset_type()
            else:
                self.dataset_type = dataset_type
            
            logger.info(f"Dataset type detected: {self.dataset_type}")
            
            # Initialize hash centers (Requirement 6.1, 6.5, 10.5)
            self._initialize_hash_centers()
    
    def _detect_dataset_type(self) -> str:
        """
        Auto-detect dataset type from label format.
        
        Returns:
            'single-label' if labels are integers
            'multi-label' if labels are arrays/vectors
        """
        # Sample the first label to determine type
        _, first_label = self.imgs[0]
        
        # Check if label is an integer (single-label) or array/vector (multi-label)
        if isinstance(first_label, (int, np.integer)):
            return 'single-label'
        elif isinstance(first_label, (np.ndarray, torch.Tensor, list)):
            return 'multi-label'
        else:
            # Default to single-label if type is unclear
            logger.warning(
                f"Unable to auto-detect dataset type from label type {type(first_label)}. "
                f"Defaulting to 'single-label'"
            )
            return 'single-label'
    
    def _initialize_hash_centers(self):
        """Initialize hash centers through generation or loading."""
        generator = HashCenterGenerator(
            num_classes=self.num_classes,
            hash_bit=self.hash_bit,
            method=self.hash_method
        )
        
        # Load from file if path provided (Requirement 10.5)
        if self.hash_centers_path is not None:
            logger.info(f"Loading hash centers from {self.hash_centers_path}")
            self.class_hash_centers = generator.load(self.hash_centers_path)
            
            # Validate loaded hash centers if requested
            if self.validate_hash_centers:
                from .validator import HashCenterValidator
                HashCenterValidator.validate(self.class_hash_centers, log_stats=True)
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
                    'dataset_type': self.dataset_type
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
            IndexError: If class index is out of range
            ValueError: If multi-label sample has no active labels
        """
        if self.dataset_type == 'single-label':
            # Direct assignment for single-label (Requirement 3.2, 3.4)
            if not isinstance(label, (int, np.integer)):
                label = int(label)
            
            # Validate class index (Requirement 3.5)
            if label < 0 or label >= self.num_classes:
                raise IndexError(
                    f"Class index {label} out of range for {self.num_classes} classes"
                )
            
            return self.class_hash_centers[label]
        
        elif self.dataset_type == 'multi-label':
            # Compute centroid for multi-label (Requirement 4.1, 4.2, 4.6)
            # Convert label to tensor if needed
            if isinstance(label, np.ndarray):
                label_vector = torch.from_numpy(label).float()
            elif isinstance(label, list):
                label_vector = torch.tensor(label, dtype=torch.float32)
            else:
                label_vector = label.float()
            
            # Calculate centroid using CentroidCalculator
            return CentroidCalculator.calculate(label_vector, self.class_hash_centers)
        
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, Union[int, torch.Tensor]], 
                                                Tuple[torch.Tensor, Union[int, torch.Tensor], torch.Tensor]]:
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
        
        # Return format based on hash center configuration
        if not self.enable_hash_centers:
            # Backward compatible: return (image, label) (Requirement 8.4, 8.5)
            return img, target
        else:
            # Get hash center for this sample (Requirement 3.3, 8.6)
            hash_center = self._get_hash_center_for_sample(target)
            return img, target, hash_center
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.imgs)
