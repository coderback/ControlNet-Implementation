"""
Conditioning input embedding for ControlNet.

Implements the paper's small convolutional network (ControlNet, arXiv 2302.05543) that maps a
pixel-space condition image (e.g. a 512x512x3 Canny edge map) into a feature map matching the
diffusion model's latent grid (64x64 x block_out_channels[0]), which is then added to the output
of the ControlNet's `conv_in`.

Module names (`conv_in`, `blocks`, `conv_out`) mirror `diffusers.ControlNetConditioningEmbedding`
so state_dict keys map 1:1 (see tests/test_parity_with_diffusers.py).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .zero_conv import zero_module


class ControlNetConditioningEmbedding(nn.Module):
    """Pixel-space condition -> latent-resolution feature map.

    A 3x3 conv stem, then `len(block_out_channels) - 1` (3x3 conv, 3x3 stride-2 conv) pairs that
    downsample 8x total (512 -> 64 for the defaults), then a final zero-initialised 3x3 conv that
    projects to `conditioning_embedding_channels` (the U-Net's block_out_channels[0]).
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding
