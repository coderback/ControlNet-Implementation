import torch
import torch.nn as nn
from src.models.zero_conv import ZeroConv2d
from src.models.condition_encoder import CannyEncoder
from src.models.controlnet import ControlNet


class DownBlockStub(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # Pass-through so we can test residual wiring
        return x


class MidBlockStub(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # Pass-through for simplicity
        return x


class SimpleUNetStub:
    class Config:
        block_out_channels = [64]
    def __init__(self):
        self.config = SimpleUNetStub.Config()
        self.down_blocks = [DownBlockStub()]
        self.mid_block = MidBlockStub()
    def parameters(self):
        # No trainable parameters for the stub in this test
        return []


def test_zero_conv_sanity():
    zconv = ZeroConv2d(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 8, 8)
    y = zconv(x)
    assert y.shape == x.shape
    # With zero-initialized weights and bias, output should be exactly zero
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_condition_encoder_shapes():
    encoder = CannyEncoder(output_channels=320)
    x = torch.randn(1, 1, 32, 32)
    y = encoder(x)
    assert y.shape == (1, 320, 2, 2)


def test_controlnet_sanity():
    fake_unet = SimpleUNetStub()
    controlnet = ControlNet(unet=fake_unet, condition_type="canny", condition_channels=1)

    # Sample latent input matching the first block_out_channels[0]
    sample = torch.randn(1, 64, 2, 2)
    timestep = torch.tensor(1)
    enc_hidden = torch.randn(1, 768)
    condition = torch.randn(1, 1, 32, 32)  # 1-channel condition input

    out_sample, residuals, mid_residual = controlnet(
        sample,
        timestep,
        enc_hidden,
        condition,
        return_controlnet_outputs=True
    )

    # Basic sanity checks
    assert out_sample.shape == sample.shape
    assert isinstance(residuals, list) and len(residuals) == len(fake_unet.down_blocks)
    assert residuals[0].shape == (1, 64, 2, 2)
    assert mid_residual.shape == (1, 64, 1, 1)

    # Residuals should be zeros due to zero-initialized convolutions
    assert torch.allclose(residuals[0], torch.zeros_like(residuals[0]), atol=1e-6)
    assert torch.allclose(mid_residual, torch.zeros_like(mid_residual), atol=1e-6)
