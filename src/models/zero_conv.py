"""
Zero-initialized layers for ControlNet.

ControlNet connects its trainable encoder copy to the frozen U-Net through 1x1 convolutions whose
weights and bias start at zero, so at the first training step the ControlNet contributes nothing and
the pretrained model is protected. As training proceeds these layers learn to inject control.

Two forms are provided:
  - `zero_module(conv)`: zero a module's parameters *in place*. Used for the model's
    `controlnet_down_blocks` / `controlnet_mid_block` so the module is a plain `nn.Conv2d` and its
    state_dict keys match `diffusers.ControlNetModel` exactly (required for the parity test).
  - `ZeroConv2d`: a small wrapper kept for the conceptual zero-conv sanity test.
"""

import torch
import torch.nn as nn


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out all parameters of a module in place and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ZeroConv2d(nn.Module):
    """A 1x1 convolution with weight and bias initialized to zero."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        zero_module(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
