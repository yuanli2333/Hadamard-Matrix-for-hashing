"""
Training script for ReIDAC-9 dataset with HashCenterDataset

This script demonstrates end-to-end training with automatic hash center generation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Import HashCenterDataset
from data_list import HashCenterDataset
from hash_center import HashCenterValidator

# Configuration
class Config:
    # Dataset
    num_classes = 9
    hash_bit = 16  # Must be power of 2 for Hadamard
    hash_method = 'hadamard'  # or 'random'
    
    # Training
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    num_workers = 4
    
    # Paths
    train_list = './reidac9_data/reidac9_list_train.txt'
    test_list = './reidac9_data/reidac9_list_test.txt'
    hash_centers_path = './reidac9_data/reidac9_hash_centers.pkl'
    checkpoint_dir = './checkpoints'
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_transforms():
    """Get data transforms for train and test."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def create_datasets(config):
    """Create train and test datasets."""
    train_transform, test_transform = get_transforms()
    
    # Load image lists
    with open(config.train_list, 'r') as f:
        train_image_list = f.readlines()
    
    with open(config.test_list, 'r') as f:
        test_image_list = f.readlines()
    
    print(f"Train images: {len(train_image_list):,}")
    print(f"Test images: {len(test_image_list):,}")
    
    # Create train dataset with hash centers
    train_dataset = HashCenterDataset(
        image_list=train_image_list,
        labels=None,
        transform=train_transform,
        num_classes=config.num_classes,
        hash_bit=config.hash_bit,
        hash_method=config.hash_method,
        dataset_type='single-label',
        dataset_name='reidac9',
        enable_hash_centers=True,
        validate_hash_centers=True,
        save_hash_centers=True,
        save_path=config.hash_centers_path
    )
    
    # Create test dataset (reuse same hash centers)
    test_dataset = HashCenterDataset(
        image_list=test_image_list,
        labels=None,
        transform=test_transform,
        num_classes=config.num_classes,
        hash_bit=config.hash_bit,
        hash_method=config.hash_method,
        dataset_type='single-label',
        dataset_name='reidac9',
        hash_centers_path=config.hash_centers_path,  # Load from file
        enable_hash_centers=True,
        validate_hash_centers=False
    )
    
    return train_dataset, test_dataset

class HashingModel(nn.Module):
    """Deep hashing model with ResNet backbone."""
    def __init__(self, hash_bit, pretrained=True):
        super().__init__()
        # Use ResNet18 as backbone
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.hash_layer = nn.Linear(512, hash_bit)
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        hash_codes = torch.tanh(self.hash_layer(features))
        return hash_codes

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, hash_centers in pbar:
        images = images.to(device)
        hash_centers = hash_centers.to(device)
        
        # Forward pass
        predicted_hash = model(images)
        loss = criterion(predicted_hash, hash_centers)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, labels, hash_centers in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            hash_centers = hash_centers.to(device)
            
            predicted_hash = model(images)
            loss = criterion(predicted_hash, hash_centers)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    config = Config()
    
    print("=" * 60)
    print("ReIDAC-9 Deep Hashing Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Classes: {config.num_classes}")
    print(f"  Hash bit: {config.hash_bit}")
    print(f"  Hash method: {config.hash_method}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Device: {config.device}")
    
    # Create datasets
    print("\n" + "=" * 60)
    print("Creating Datasets")
    print("=" * 60)
    train_dataset, test_dataset = create_datasets(config)
    
    # Validate hash centers
    print("\n" + "=" * 60)
    print("Hash Center Quality")
    print("=" * 60)
    stats = HashCenterValidator.validate(train_dataset.class_hash_centers)
    print(f"  Min distance: {stats['min_distance']}")
    print(f"  Max distance: {stats['max_distance']}")
    print(f"  Avg distance: {stats['avg_distance']:.2f}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)
    model = HashingModel(config.hash_bit, pretrained=True)
    model = model.to(config.device)
    print(f"  Model: ResNet18 + Hash Layer")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_loss = float('inf')
    history = {'train_loss': [], 'test_loss': []}
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # Evaluate
        test_loss = evaluate(model, test_loader, criterion, config.device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}")
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            Path(config.checkpoint_dir).mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, f'{config.checkpoint_dir}/best_model.pth')
            print(f"  ✓ Saved best model (test loss: {test_loss:.4f})")
    
    # Save final model
    torch.save(model.state_dict(), f'{config.checkpoint_dir}/final_model.pth')
    
    # Save training history
    with open(f'{config.checkpoint_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Best test loss: {best_loss:.4f}")
    print(f"  Model saved to: {config.checkpoint_dir}/")

if __name__ == "__main__":
    main()
