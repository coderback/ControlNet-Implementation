"""
HEADLINE TEST: prove the hand-written ControlNet matches diffusers' reference implementation.

The reference `diffusers.ControlNetModel` is used ONLY here, as a known-correct oracle — never
imported by src/. We build both from the same SD-shaped U-Net, RANDOMIZE the reference's weights
(the control path is zero-initialised, so without this the residuals would be a trivial 0 == 0),
copy the reference state_dict into our hand-written model (architecture is 1:1, so keys map), feed
identical inputs, and assert every down-block residual and the mid residual match to ~1e-4.

Runs on CPU in seconds.
"""

import torch
from diffusers import ControlNetModel, UNet2DConditionModel

from src.models.controlnet import ControlNet


def _small_unet() -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        norm_num_groups=16,
    )


def test_controlnet_residuals_match_diffusers():
    torch.manual_seed(0)
    unet = _small_unet()

    reference = ControlNetModel.from_unet(unet)

    # Randomize every float parameter so the zero-init control path produces non-trivial residuals.
    sd = reference.state_dict()
    for k, v in sd.items():
        if v.is_floating_point():
            sd[k] = torch.randn_like(v) * 0.1
    reference.load_state_dict(sd)

    mine = ControlNet(unet)
    mine.load_state_dict(reference.state_dict())  # 1:1 key mapping by design

    reference.eval()
    mine.eval()

    sample = torch.randn(1, 4, 16, 16)
    timestep = torch.tensor([10])
    encoder_hidden_states = torch.randn(1, 4, 32)
    cond = torch.rand(1, 3, 128, 128)

    with torch.no_grad():
        ref_down, ref_mid = reference(
            sample, timestep, encoder_hidden_states, controlnet_cond=cond, return_dict=False
        )
        my_down, my_mid = mine(
            sample, timestep, encoder_hidden_states, controlnet_cond=cond, return_dict=False
        )

    assert len(my_down) == len(ref_down), f"{len(my_down)} vs {len(ref_down)} down residuals"
    for i, (a, b) in enumerate(zip(my_down, ref_down)):
        assert a.shape == b.shape, f"down[{i}] shape {a.shape} vs {b.shape}"
        assert torch.allclose(a, b, atol=1e-4), f"down[{i}] max diff {(a - b).abs().max().item()}"
    assert torch.allclose(my_mid, ref_mid, atol=1e-4), f"mid max diff {(my_mid - ref_mid).abs().max().item()}"
