"""
ControlNet block implementation.

This implements the core ControlNet architecture where a neural block is enhanced
with conditional control through a trainable copy connected via zero convolutions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .zero_conv import ZeroConv2d


class ControlNetBlock(nn.Module):
    """
    Core ControlNet block implementation.
    
    This takes a pretrained neural block F(x; Θ) and creates:
    1. A locked (frozen) copy of the original block
    2. A trainable copy that learns to process conditioning
    3. Zero convolutions to connect them
    
    The output is: yc = F(x; Θ) + Z(F(x + Z(c; Θz1); Θc); Θz2)
    """
    
    def __init__(
        self,
        block: nn.Module,
        condition_channels: int,
        block_channels: int,
        make_trainable_copy: bool = True
    ):
        """
        Args:
            block: The original pretrained neural block to control
            condition_channels: Number of channels in the conditioning input
            block_channels: Number of channels in the block's input/output
            make_trainable_copy: Whether to create a trainable copy (True) or use the original
        """
        super().__init__()
        
        # Store the original block (will be frozen)
        self.original_block = block
        
        # Create trainable copy if requested
        if make_trainable_copy:
            # Deep copy the block for training
            self.trainable_copy = self._make_trainable_copy(block)
        else:
            self.trainable_copy = block
        
        # Zero convolution layers
        # Z(c; Θz1) - processes the conditioning input
        self.zero_conv_condition = ZeroConv2d(
            in_channels=condition_channels,
            out_channels=block_channels,
            kernel_size=1
        )
        
        # Z(·; Θz2) - processes the trainable copy output
        self.zero_conv_output = ZeroConv2d(
            in_channels=block_channels,
            out_channels=block_channels,
            kernel_size=1
        )
        
        # Freeze the original block
        self._freeze_original_block()
    
    def _make_trainable_copy(self, block: nn.Module) -> nn.Module:
        """Create a trainable copy of the block."""
        import copy
        trainable_copy = copy.deepcopy(block)
        
        # Ensure all parameters are trainable
        for param in trainable_copy.parameters():
            param.requires_grad_(True)
        
        return trainable_copy
    
    def _freeze_original_block(self):
        """Freeze all parameters in the original block."""
        for param in self.original_block.parameters():
            param.requires_grad_(False)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass implementing the ControlNet equation:
        yc = F(x; Θ) + Z(F(x + Z(c; Θz1); Θc); Θz2)
        
        Args:
            x: Input feature map
            condition: Conditioning tensor (can be None)
        
        Returns:
            Controlled output feature map
        """
        # Original block output: F(x; Θ)
        original_output = self.original_block(x)
        
        if condition is None:
            # No conditioning, return original output
            return original_output
        
        # Process conditioning: Z(c; Θz1)
        processed_condition = self.zero_conv_condition(condition)
        
        # Add conditioning to input: x + Z(c; Θz1)
        conditioned_input = x + processed_condition
        
        # Trainable copy with conditioned input: F(x + Z(c; Θz1); Θc)
        trainable_output = self.trainable_copy(conditioned_input)
        
        # Process trainable output: Z(F(x + Z(c; Θz1); Θc); Θz2)
        processed_trainable = self.zero_conv_output(trainable_output)
        
        # Final output: yc = F(x; Θ) + Z(F(x + Z(c; Θz1); Θc); Θz2)
        controlled_output = original_output + processed_trainable
        
        return controlled_output


class ControlNetResBlock(ControlNetBlock):
    """
    ControlNet block specifically designed for ResNet-style blocks.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_channels: int,
        stride: int = 1,
        groups: int = 1
    ):
        # Create a simple ResNet block
        resnet_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, groups=groups),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=groups),
            nn.GroupNorm(32, out_channels)
        )
        
        super().__init__(
            block=resnet_block,
            condition_channels=condition_channels,
            block_channels=out_channels
        )
        
        # Skip connection
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get controlled output
        controlled_output = super().forward(x, condition)
        
        # Add skip connection
        skip = self.skip_connection(x)
        
        return controlled_output + skip


class ControlNetAttentionBlock(ControlNetBlock):
    """
    ControlNet block for attention mechanisms.
    """
    
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        num_heads: int = 8,
        head_dim: int = 64
    ):
        # Create a simple attention block
        attention_block = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        super().__init__(
            block=attention_block,
            condition_channels=condition_channels,
            block_channels=channels
        )
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Reshape for attention if needed
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
        
        if condition is not None:
            condition_flat = condition.view(b, -1, h * w).transpose(1, 2)
        else:
            condition_flat = None
        
        # Apply controlled attention
        controlled_flat = super().forward(x_flat, condition_flat)
        
        # Reshape back
        controlled_output = controlled_flat.transpose(1, 2).view(b, c, h, w)
        
        return controlled_output