"""
Tiny end-to-end smoke test: ControlNet + frozen mini-U-Net, one forward/backward on random
tensors, asserting (a) it runs, (b) only ControlNet params receive gradients, (c) loss is finite.
Catches the class of bug that broke the original repo (wrong block signature, residual count
mismatch, embeddings not plumbed through) for $0, on CPU.
"""

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel

from src.models.controlnet import ControlNet


def test_one_training_step_runs_and_only_controlnet_learns():
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=2,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32, norm_num_groups=16,
    )
    for p in unet.parameters():
        p.requires_grad_(False)
    controlnet = ControlNet.from_unet(unet)

    sample = torch.randn(1, 4, 16, 16)
    t = torch.tensor([10])
    ehs = torch.randn(1, 4, 32)
    cond = torch.rand(1, 3, 128, 128)
    noise = torch.randn_like(sample)

    down, mid = controlnet(sample, t, ehs, controlnet_cond=cond, return_dict=False)
    pred = unet(sample, t, ehs, down_block_additional_residuals=down,
                mid_block_additional_residual=mid).sample
    loss = F.mse_loss(pred, noise)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(p.grad is None for p in unet.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() >= 0 for p in controlnet.parameters())
