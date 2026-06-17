"""Zero-convolution sanity: a freshly-built zero conv outputs exactly zero."""

import torch

from src.models.zero_conv import ZeroConv2d, zero_module
import torch.nn as nn


def test_zero_conv_outputs_zero():
    zconv = ZeroConv2d(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 8, 8)
    y = zconv(x)
    assert y.shape == x.shape
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_zero_module_zeros_params():
    conv = zero_module(nn.Conv2d(4, 4, kernel_size=1))
    assert all(torch.count_nonzero(p) == 0 for p in conv.parameters())
    assert torch.allclose(conv(torch.randn(1, 4, 8, 8)), torch.zeros(1, 4, 8, 8), atol=1e-6)
