"""
Dataset implementation for ControlNet training.

This provides data loading for various conditioning types (Canny, depth, pose, etc.)
along with corresponding images and text prompts.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..models.condition_encoder import preprocess_canny, preprocess_depth, preprocess_pose


class ControlNetDataset(Dataset):
    """
    Dataset for ControlNet training.
    
    Supports various conditioning types:
    - Canny edges
    - Depth maps  
    - Human poses
    - Segmentation maps
    - Surface normals
    - User scribbles
    """
    
    def __init__(
        self,
        data_root: str,
        condition_type: str = "canny",
        image_size: int = 512,
        split: str = "train",
        max_samples: Optional[int] = None,
        augment: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing the dataset
            condition_type: Type of conditioning ('canny', 'depth', 'pose', etc.)
            image_size: Target image size for training
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load (for debugging)
            augment: Whether to apply data augmentation
        """
        self.data_root = Path(data_root)
        self.condition_type = condition_type
        self.image_size = image_size
        self.split = split
        self.augment = augment
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        # Setup image transforms
        self.transforms = self._setup_transforms()
        
        print(f"Loaded {len(self.samples)} samples for {condition_type} conditioning")
    
    def _load_samples(self) -> List[Dict]:
        """Load dataset samples from metadata file."""
        metadata_file = self.data_root / f"{self.split}.json"
        
        if not metadata_file.exists():
            # If no metadata file, scan directories
            return self._scan_directories()
        
        with open(metadata_file, 'r') as f:
            samples = json.load(f)
        
        return samples
    
    def _scan_directories(self) -> List[Dict]:
        """Scan directories to create sample list."""
        samples = []
        
        # Expected directory structure:
        # data_root/
        #   images/
        #   conditions/
        #   prompts.json (optional)
        
        image_dir = self.data_root / "images"
        condition_dir = self.data_root / "conditions" / self.condition_type
        prompts_file = self.data_root / "prompts.json"
        
        # Load prompts if available
        prompts = {}
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                prompts = json.load(f)
        
        # Scan for matching image and condition pairs
        for img_path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
            # Look for corresponding condition file
            condition_path = condition_dir / f"{img_path.stem}.png"
            if not condition_path.exists():
                condition_path = condition_dir / f"{img_path.stem}.jpg"
            
            if condition_path.exists():
                sample = {
                    "image_path": str(img_path),
                    "condition_path": str(condition_path),
                    "prompt": prompts.get(img_path.stem, "")
                }
                samples.append(sample)
        
        return samples
    
    def _setup_transforms(self) -> A.Compose:
        """Setup image transforms for training."""
        transforms = []
        
        # Resize
        transforms.append(A.Resize(self.image_size, self.image_size))
        
        if self.augment and self.split == "train":
            # Data augmentation
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RandomGamma(p=0.3),
            ])
        
        # Normalization for Stable Diffusion
        transforms.extend([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
        
        return A.Compose(
            transforms,
            additional_targets={"condition": "image"}
        )
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image."""
        image = Image.open(path).convert("RGB")
        return np.array(image)
    
    def _load_condition(self, path: str) -> np.ndarray:
        """Load and preprocess conditioning image."""
        if self.condition_type == "canny":
            # For Canny, we might store the original image and compute edges
            if Path(path).suffix in ['.jpg', '.png']:
                image = cv2.imread(path)
                return preprocess_canny(image)
            else:
                # Pre-computed Canny edges
                condition = Image.open(path).convert("L")
                return np.array(condition)[..., None]
        
        elif self.condition_type == "depth":
            # Depth maps are typically stored as grayscale or .npy files
            if path.endswith('.npy'):
                depth = np.load(path)
            else:
                depth = np.array(Image.open(path).convert("L"))
            return preprocess_depth(depth)
        
        elif self.condition_type == "pose":
            # Pose images are typically RGB visualizations
            condition = Image.open(path).convert("RGB")
            return np.array(condition)
        
        else:
            # Generic RGB condition
            condition = Image.open(path).convert("RGB")
            return np.array(condition)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load image and condition
        image = self._load_image(sample["image_path"])
        condition = self._load_condition(sample["condition_path"])
        
        # Apply transforms
        transformed = self.transforms(image=image, condition=condition)
        
        return {
            "image": transformed["image"],
            "condition": transformed["condition"],
            "prompt": sample.get("prompt", ""),
            "image_path": sample["image_path"],
            "condition_path": sample["condition_path"]
        }


class SyntheticCannyDataset(Dataset):
    """
    Synthetic dataset that generates Canny edges from regular images.
    Useful for quick experimentation without pre-processed conditioning data.
    """
    
    def __init__(
        self,
        image_dir: str,
        image_size: int = 512,
        low_threshold: int = 100,
        high_threshold: int = 200,
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.split = split
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
        
        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]
        
        self.transforms = self._setup_transforms()
        
        print(f"Created synthetic Canny dataset with {len(self.image_paths)} images")
    
    def _setup_transforms(self) -> A.Compose:
        """Setup transforms."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5) if self.split == "train" else A.NoOp(),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ], additional_targets={"condition": "image"})
    
    def _generate_canny(self, image: np.ndarray) -> np.ndarray:
        """Generate Canny edges from image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # Convert to 3-channel for consistency
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges_rgb
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_path = self.image_paths[idx]
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Generate Canny edges
        condition = self._generate_canny(image)
        
        # Apply transforms
        transformed = self.transforms(image=image, condition=condition)
        
        return {
            "image": transformed["image"],
            "condition": transformed["condition"],
            "prompt": "",  # No prompts in this simple dataset
            "image_path": str(image_path)
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader with appropriate settings."""
    
    def collate_fn(batch):
        """Custom collate function."""
        images = torch.stack([item["image"] for item in batch])
        conditions = torch.stack([item["condition"] for item in batch])
        prompts = [item["prompt"] for item in batch]
        
        return {
            "image": images,
            "condition": conditions,
            "prompt": prompts
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True  # Important for stable training
    )


# Dataset factory function
def create_dataset(
    data_root: str,
    condition_type: str,
    image_size: int = 512,
    split: str = "train",
    **kwargs
) -> Dataset:
    """Factory function to create datasets."""
    
    if condition_type == "synthetic_canny":
        return SyntheticCannyDataset(
            image_dir=data_root,
            image_size=image_size,
            split=split,
            **kwargs
        )
    else:
        return ControlNetDataset(
            data_root=data_root,
            condition_type=condition_type,
            image_size=image_size,
            split=split,
            **kwargs
        )