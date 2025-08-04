"""
Zero Convolution layers for ControlNet.

These are 1x1 convolution layers with weights and bias initialized to zero.
They allow gradual learning of control conditions without affecting the pretrained model initially.
"""

import torch
import torch.nn as nn
from typing import Optional


class ZeroConv2d(nn.Module):
    """
    Zero-initialized 2D convolution layer.
    
    This is a 1x1 convolution with both weight and bias initialized to zeros.
    During the first training step, this layer outputs zeros, protecting the
    pretrained model from harmful noise.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        # Zero initialization - this is the key innovation
        nn.init.zeros_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ZeroConv1d(nn.Module):
    """Zero-initialized 1D convolution layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        nn.init.zeros_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ZeroLinear(nn.Module):
    """Zero-initialized linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def make_zero_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int = 0
) -> ZeroConv2d:
    """Factory function to create zero convolution layers."""
    return ZeroConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )