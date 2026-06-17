"""
ControlNet — a faithful, parity-verified reimplementation.

ControlNet (arXiv 2302.05543) adds spatial control to a frozen text-to-image diffusion U-Net by:
  1. cloning the U-Net's encoder (down blocks) + middle block into a trainable copy,
  2. mapping a pixel-space condition (e.g. Canny edges) into the latent grid via a small
     conditioning network, added to the trainable copy's `conv_in` output,
  3. connecting every intermediate feature of the copy back to the frozen U-Net's decoder through
     zero-initialised 1x1 convolutions (so training starts as an exact no-op).

We hand-write ControlNet's wiring (assembly, forward, the zero convs, the conditioning network,
and `from_unet`) but reuse Stable Diffusion's own block primitives (`get_down_block`,
`UNetMidBlock2DCrossAttn`, `Timesteps`, `TimestepEmbedding`) — these are the base model, not
ControlNet's contribution, and reusing them lets the output match `diffusers.ControlNetModel`
bit-for-bit given equal weights (tests/test_parity_with_diffusers.py).

The forward returns `(down_block_res_samples, mid_block_res_sample)` that plug directly into a
frozen `UNet2DConditionModel(..., down_block_additional_residuals=, mid_block_additional_residual=)`.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn, get_down_block
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from .conditioning import ControlNetConditioningEmbedding
from .zero_conv import zero_module


@dataclass
class ControlNetOutput:
    down_block_res_samples: Tuple[torch.Tensor, ...]
    mid_block_res_sample: torch.Tensor


class ControlNet(nn.Module):
    """Hand-written ControlNet built to mirror `diffusers.ControlNetModel` structurally.

    Construct from an existing U-Net so the encoder geometry always matches the base model:

        controlnet = ControlNet(unet)                       # structure only (zero-init control path)
        controlnet = ControlNet.from_unet(unet)             # + copy pretrained encoder weights (training)
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()
        c = unet.config

        block_out_channels: Tuple[int, ...] = tuple(c.block_out_channels)
        down_block_types: Tuple[str, ...] = tuple(c.down_block_types)
        layers_per_block: int = c.layers_per_block

        transformer_layers_per_block = getattr(c, "transformer_layers_per_block", 1)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        num_attention_heads = c.num_attention_heads or c.attention_head_dim
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        only_cross_attention = c.only_cross_attention
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        # --- input conv ---
        self.conv_in = nn.Conv2d(c.in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # --- time embedding ---
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], c.flip_sin_to_cos, c.freq_shift)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim, act_fn=c.act_fn)

        # --- conditioning embedding (pixel-space condition -> latent grid) ---
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        # --- trainable encoder copy + per-feature zero convs ---
        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        # zero conv for the post-conv_in sample
        self.controlnet_down_blocks.append(zero_module(nn.Conv2d(output_channel, output_channel, kernel_size=1)))

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=c.norm_eps,
                resnet_act_fn=c.act_fn,
                resnet_groups=c.norm_num_groups,
                cross_attention_dim=c.cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                attention_head_dim=c.attention_head_dim if isinstance(c.attention_head_dim, int) else c.attention_head_dim[i],
                downsample_padding=c.downsample_padding,
                use_linear_projection=c.use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=c.upcast_attention,
                resnet_time_scale_shift=c.resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                self.controlnet_down_blocks.append(zero_module(nn.Conv2d(output_channel, output_channel, kernel_size=1)))
            if not is_final_block:
                self.controlnet_down_blocks.append(zero_module(nn.Conv2d(output_channel, output_channel, kernel_size=1)))

        # --- middle block + its zero conv ---
        mid_block_channel = block_out_channels[-1]
        self.controlnet_mid_block = zero_module(nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1))
        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=c.norm_eps,
            resnet_act_fn=c.act_fn,
            output_scale_factor=c.mid_block_scale_factor,
            resnet_time_scale_shift=c.resnet_time_scale_shift,
            cross_attention_dim=c.cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=c.norm_num_groups,
            use_linear_projection=c.use_linear_projection,
            upcast_attention=c.upcast_attention,
        )

    @classmethod
    def from_unet(cls, unet: UNet2DConditionModel, copy_weights: bool = True, **kwargs) -> "ControlNet":
        """Build a ControlNet from a U-Net, optionally copying the pretrained encoder weights into
        the trainable copy (as the paper requires — the copy starts from the pretrained model)."""
        controlnet = cls(unet, **kwargs)
        if copy_weights:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())
        return controlnet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[List[torch.Tensor], torch.Tensor]]:
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and timesteps.dim() == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process: conv_in + conditioning
        sample = self.conv_in(sample)
        sample = sample + self.controlnet_cond_embedding(controlnet_cond)

        # 3. down (collect every intermediate feature, seeded with the post-conv_in sample)
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        # 5. zero convs
        controlnet_down_block_res_samples = ()
        for res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            controlnet_down_block_res_samples += (controlnet_block(res_sample),)
        down_block_res_samples = controlnet_down_block_res_samples
        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        down_block_res_samples = [s * conditioning_scale for s in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)
        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )
