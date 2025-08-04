"""
Conditioning encoder networks for ControlNet.

These networks encode various types of conditioning inputs (images, poses, etc.)
into feature representations that can be used to control diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union
import numpy as np


class ConditionEncoder(nn.Module):
    """
    Base class for conditioning encoders.
    
    The encoder takes a conditioning input (e.g., Canny edge, depth map, pose)
    and converts it to a feature representation compatible with the diffusion model.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 320,  # Stable Diffusion's first layer channels
        hidden_channels: list = [16, 32, 64, 128]
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Build encoder layers
        layers = []
        in_ch = input_channels
        
        for hidden_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, hidden_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ])
            in_ch = hidden_ch
        
        # Final layer to match output channels
        layers.append(
            nn.Conv2d(in_ch, output_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Encode conditioning input to feature representation.
        
        Args:
            condition: Input condition tensor [B, C, H, W]
        
        Returns:
            Encoded features [B, output_channels, H//8, W//8]
        """
        return self.encoder(condition)


class CannyEncoder(ConditionEncoder):
    """Encoder for Canny edge conditioning."""
    
    def __init__(self, output_channels: int = 320):
        super().__init__(
            input_channels=1,  # Canny edges are grayscale
            output_channels=output_channels,
            hidden_channels=[16, 32, 64, 128]
        )


class DepthEncoder(ConditionEncoder):
    """Encoder for depth map conditioning."""
    
    def __init__(self, output_channels: int = 320):
        super().__init__(
            input_channels=1,  # Depth maps are single channel
            output_channels=output_channels,
            hidden_channels=[16, 32, 64, 128]
        )


class PoseEncoder(ConditionEncoder):
    """Encoder for human pose conditioning."""
    
    def __init__(self, output_channels: int = 320):
        super().__init__(
            input_channels=3,  # RGB pose keypoints
            output_channels=output_channels,
            hidden_channels=[16, 32, 64, 128]
        )


class SegmentationEncoder(ConditionEncoder):
    """Encoder for segmentation map conditioning."""
    
    def __init__(self, output_channels: int = 320, num_classes: int = 150):
        # Convert segmentation to one-hot or use embedding
        super().__init__(
            input_channels=3,  # RGB segmentation visualization
            output_channels=output_channels,
            hidden_channels=[16, 32, 64, 128]
        )


class NormalEncoder(ConditionEncoder):
    """Encoder for surface normal conditioning."""
    
    def __init__(self, output_channels: int = 320):
        super().__init__(
            input_channels=3,  # Normal maps are RGB
            output_channels=output_channels,
            hidden_channels=[16, 32, 64, 128]
        )


class ScribbleEncoder(ConditionEncoder):
    """Encoder for user scribble conditioning."""
    
    def __init__(self, output_channels: int = 320):
        super().__init__(
            input_channels=1,  # Scribbles are typically binary/grayscale
            output_channels=output_channels,
            hidden_channels=[16, 32, 64, 128]
        )


class MultiScaleConditionEncoder(nn.Module):
    """
    Multi-scale condition encoder that produces features at multiple resolutions.
    This is useful for integrating with different levels of the U-Net.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels_list: list = [320, 640, 1280, 1280],  # SD U-Net channels
        scales: list = [8, 16, 32, 64]  # Downsampling factors
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels_list = output_channels_list
        self.scales = scales
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        
        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList()
        for i, (out_ch, scale) in enumerate(zip(output_channels_list, scales)):
            # Calculate how much downsampling we need
            downsample_layers = []
            current_scale = 1
            in_ch = 64
            
            while current_scale < scale:
                downsample_layers.extend([
                    nn.Conv2d(in_ch, min(in_ch * 2, 512), 4, stride=2, padding=1),
                    nn.ReLU()
                ])
                in_ch = min(in_ch * 2, 512)
                current_scale *= 2
            
            # Final projection to target channels
            downsample_layers.append(
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            )
            
            self.scale_encoders.append(nn.Sequential(*downsample_layers))
    
    def forward(self, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode condition at multiple scales.
        
        Args:
            condition: Input condition [B, C, H, W]
        
        Returns:
            Dictionary mapping scale names to encoded features
        """
        # Extract shared features
        features = self.feature_extractor(condition)
        
        # Encode at each scale
        multi_scale_features = {}
        for i, (scale, encoder) in enumerate(zip(self.scales, self.scale_encoders)):
            scale_features = encoder(features)
            multi_scale_features[f"scale_{scale}"] = scale_features
        
        return multi_scale_features


def create_condition_encoder(
    condition_type: str,
    output_channels: int = 320,
    **kwargs
) -> ConditionEncoder:
    """
    Factory function to create condition encoders.
    
    Args:
        condition_type: Type of conditioning ('canny', 'depth', 'pose', etc.)
        output_channels: Number of output channels
        **kwargs: Additional arguments for specific encoders
    
    Returns:
        Appropriate condition encoder
    """
    encoders = {
        'canny': CannyEncoder,
        'depth': DepthEncoder,
        'pose': PoseEncoder,
        'segmentation': SegmentationEncoder,
        'normal': NormalEncoder,
        'scribble': ScribbleEncoder,
        'generic': ConditionEncoder
    }
    
    if condition_type not in encoders:
        raise ValueError(f"Unknown condition type: {condition_type}")
    
    encoder_class = encoders[condition_type]
    return encoder_class(output_channels=output_channels, **kwargs)


# Preprocessing utilities for different condition types
def preprocess_canny(image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    """Preprocess image to Canny edges."""
    import cv2
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges[..., None]  # Add channel dimension


def preprocess_depth(depth: np.ndarray) -> np.ndarray:
    """Preprocess depth map."""
    # Normalize depth to [0, 1]
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth_norm[..., None]  # Add channel dimension


def preprocess_pose(pose_keypoints: np.ndarray, image_size: tuple = (512, 512)) -> np.ndarray:
    """Convert pose keypoints to pose image."""
    # This is a simplified version - real implementation would use OpenPose format
    pose_image = np.zeros((*image_size, 3), dtype=np.uint8)
    
    # Draw keypoints and connections (simplified)
    # In practice, you'd use proper pose visualization
    
    return pose_image