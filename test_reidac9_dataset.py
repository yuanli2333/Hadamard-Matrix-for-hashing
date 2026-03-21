"""
Quick test script for ReIDAC-9 dataset with HashCenterDataset
"""

import torch
from torchvision import transforms
from data_list import HashCenterDataset
from hash_center import HashCenterValidator
import matplotlib.pyplot as plt
import numpy as np

# Configuration
num_classes = 9
hash_bit = 16
train_list = './reidac9_data/reidac9_list_train.txt'
hash_centers_path = './reidac9_data/reidac9_hash_centers.pkl'

# Load class names
class_names = []
with open('./reidac9_data/reidac9_list_classes.txt', 'r') as f:
    for line in f:
        idx, name = line.strip().split()
        class_names.append(name)

print("=" * 60)
print("ReIDAC-9 Dataset Test")
print("=" * 60)
print(f"\nClasses ({len(class_names)}):")
for idx, name in enumerate(class_names):
    print(f"  {idx}: {name}")

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load image list
with open(train_list, 'r') as f:
    image_list = f.readlines()

print(f"\nTotal images: {len(image_list):,}")

# Create dataset
print("\n" + "=" * 60)
print("Creating HashCenterDataset")
print("=" * 60)

dataset = HashCenterDataset(
    image_list=image_list,
    labels=None,
    transform=transform,
    num_classes=num_classes,
    hash_bit=hash_bit,
    hash_method='hadamard',
    dataset_type='single-label',
    dataset_name='reidac9',
    enable_hash_centers=True,
    validate_hash_centers=True,
    save_hash_centers=True,
    save_path=hash_centers_path
)

print(f"\n✓ Dataset created with {len(dataset):,} samples")
print(f"✓ Hash centers shape: {dataset.class_hash_centers.shape}")

# Validate hash centers
print("\n" + "=" * 60)
print("Hash Center Quality Analysis")
print("=" * 60)

stats = HashCenterValidator.validate(dataset.class_hash_centers, log_stats=True)
print(f"\nStatistics:")
print(f"  Min Hamming distance: {stats['min_distance']}")
print(f"  Max Hamming distance: {stats['max_distance']}")
print(f"  Avg Hamming distance: {stats['avg_distance']:.2f}")
print(f"  Std Hamming distance: {stats['std_distance']:.2f}")
print(f"  Is valid: {stats['is_valid']}")

# Display hash centers
print("\n" + "=" * 60)
print("Hash Centers")
print("=" * 60)
for i in range(num_classes):
    hash_center = dataset.class_hash_centers[i]
    print(f"\n{class_names[i]:15s} (Class {i}): {hash_center.numpy()}")

# Test loading samples
print("\n" + "=" * 60)
print("Sample Data")
print("=" * 60)

for i in range(min(5, len(dataset))):
    image, label, hash_center = dataset[i]
    print(f"\nSample {i}:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label} ({class_names[label]})")
    print(f"  Hash center shape: {hash_center.shape}")
    print(f"  Hash center: {hash_center.numpy()}")
    
    # Verify hash center matches class hash center
    expected = dataset.class_hash_centers[label]
    assert torch.equal(hash_center, expected), "Hash center mismatch!"

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nYou can now run: python train_reidac9.py")
