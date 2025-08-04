"""
Main ControlNet implementation.

This implements the full ControlNet architecture that adds conditional control
to pretrained diffusion models like Stable Diffusion.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import copy

from .controlnet_block import ControlNetBlock
from .condition_encoder import ConditionEncoder, create_condition_encoder
from .zero_conv import ZeroConv2d


class ControlNet(nn.Module):
    """
    ControlNet implementation for adding conditional control to diffusion models.
    
    This creates a trainable copy of a pretrained U-Net's encoder blocks and middle block,
    connects them with zero convolutions, and allows conditioning on additional inputs.
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        condition_type: str = "canny",
        condition_channels: int = 3,
        conditioning_scale: float = 1.0
    ):
        """
        Initialize ControlNet.
        
        Args:
            unet: Pretrained U-Net model to control
            condition_type: Type of conditioning ('canny', 'depth', 'pose', etc.)
            condition_channels: Number of channels in conditioning input
            conditioning_scale: Scale factor applied to conditioning
        """
        super().__init__()
        
        self.conditioning_scale = conditioning_scale
        self.condition_type = condition_type
        
        # Store reference to original U-Net (will be frozen)
        self.unet = unet
        
        # Create condition encoder
        self.condition_encoder = create_condition_encoder(
            condition_type=condition_type,
            output_channels=unet.config.block_out_channels[0]  # Match first U-Net layer
        )
        
        # Create trainable copies of U-Net encoder blocks
        self.controlnet_down_blocks = nn.ModuleList()
        self.controlnet_mid_block = None
        
        # Zero convolutions for each level
        self.zero_convs_down = nn.ModuleList()
        self.zero_conv_mid = None
        
        self._build_controlnet_blocks()
        self._freeze_unet()
    
    def _build_controlnet_blocks(self):
        """Build ControlNet blocks from U-Net architecture."""
        # Create trainable copies of down blocks
        for i, down_block in enumerate(self.unet.down_blocks):
            # Create trainable copy
            controlnet_block = copy.deepcopy(down_block)
            self.controlnet_down_blocks.append(controlnet_block)
            
            # Create zero convolution for this level
            block_channels = self.unet.config.block_out_channels[i]
            zero_conv = ZeroConv2d(block_channels, block_channels)
            self.zero_convs_down.append(zero_conv)
        
        # Create trainable copy of middle block
        if self.unet.mid_block is not None:
            self.controlnet_mid_block = copy.deepcopy(self.unet.mid_block)
            
            # Zero convolution for middle block
            mid_channels = self.unet.config.block_out_channels[-1]
            self.zero_conv_mid = ZeroConv2d(mid_channels, mid_channels)
    
    def _freeze_unet(self):
        """Freeze all parameters in the original U-Net."""
        for param in self.unet.parameters():
            param.requires_grad_(False)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        condition: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_controlnet_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through ControlNet.
        
        Args:
            sample: Noisy input sample
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings
            condition: Conditioning input (e.g., Canny edges, depth map)
            ... (other standard U-Net arguments)
            return_controlnet_outputs: Whether to return ControlNet residuals
        
        Returns:
            If return_controlnet_outputs=False: Final denoised sample
            If return_controlnet_outputs=True: (sample, controlnet_residuals)
        """
        # Encode conditioning input
        encoded_condition = self.condition_encoder(condition)
        
        # Ensure encoded condition matches sample batch size
        if encoded_condition.shape[0] != sample.shape[0]:
            # Expand condition to match batch size (for inference with CFG)
            encoded_condition = encoded_condition.repeat(sample.shape[0] // encoded_condition.shape[0], 1, 1, 1)
        
        # Process through ControlNet blocks
        controlnet_residuals = []
        mid_residual = None
        
        # Start with the encoded condition
        hidden_states = encoded_condition
        
        # Forward through encoder blocks with conditioning
        for i, (down_block, zero_conv) in enumerate(zip(self.controlnet_down_blocks, self.zero_convs_down)):
            # Get the sample at this level (downsample as needed)
            level_sample = sample
            for _ in range(i):
                level_sample = F.avg_pool2d(level_sample, kernel_size=2, stride=2)
            
            # Resize condition to match this level if needed
            if hidden_states.shape[-2:] != level_sample.shape[-2:]:
                hidden_states = F.interpolate(
                    hidden_states, 
                    size=level_sample.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Forward through ControlNet block with conditioning
            if hasattr(down_block, 'forward'):
                # Simple block forward
                block_output = down_block(level_sample + hidden_states)
            else:
                # Handle more complex blocks
                block_output = level_sample + hidden_states
                for layer in down_block:
                    if hasattr(layer, '__call__'):
                        block_output = layer(block_output)
            
            # Apply zero convolution
            controlnet_residual = zero_conv(block_output)
            controlnet_residuals.append(controlnet_residual * self.conditioning_scale)
            
            # Update hidden states for next level (downsample)
            hidden_states = F.avg_pool2d(block_output, kernel_size=2, stride=2)
        
        # Handle middle block
        if self.controlnet_mid_block is not None and self.zero_conv_mid is not None:
            # Resize condition to match middle block size
            mid_size = (sample.shape[-2] // (2 ** len(self.controlnet_down_blocks)), 
                       sample.shape[-1] // (2 ** len(self.controlnet_down_blocks)))
            
            if hidden_states.shape[-2:] != mid_size:
                hidden_states = F.interpolate(
                    hidden_states,
                    size=mid_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            # Get sample at middle level
            mid_sample = sample
            for _ in range(len(self.controlnet_down_blocks)):
                mid_sample = F.avg_pool2d(mid_sample, kernel_size=2, stride=2)
            
            # Forward through middle block
            if hasattr(self.controlnet_mid_block, 'forward'):
                mid_output = self.controlnet_mid_block(mid_sample + hidden_states)
            else:
                mid_output = mid_sample + hidden_states
                for layer in self.controlnet_mid_block:
                    if hasattr(layer, '__call__'):
                        mid_output = layer(mid_output)
            
            mid_residual = self.zero_conv_mid(mid_output) * self.conditioning_scale
        
        if return_controlnet_outputs:
            return sample, controlnet_residuals, mid_residual
        
        # ControlNet should not run the full U-Net - it only provides residuals
        # The caller (training loop or pipeline) handles the U-Net forward pass
        raise ValueError(
            "ControlNet forward with return_controlnet_outputs=False is not supported. "
            "Use return_controlnet_outputs=True and handle U-Net forward in the caller."
        )


class ControlNetPipeline(nn.Module):
    """
    Complete pipeline combining ControlNet with Stable Diffusion.
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        controlnet: ControlNet,
        vae: Any = None,
        text_encoder: Any = None,
        tokenizer: Any = None,
        scheduler: Any = None
    ):
        super().__init__()
        
        self.unet = unet
        self.controlnet = controlnet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        condition_image: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate images with ControlNet conditioning.
        
        Args:
            prompt: Text prompt
            condition_image: Conditioning image (e.g., Canny edges)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            conditioning_scale: ControlNet conditioning scale
            height, width: Output image dimensions
            generator: Random generator for reproducibility
        
        Returns:
            Generated image tensor
        """
        device = next(self.controlnet.parameters()).device
        
        # Encode text prompt
        if self.text_encoder is not None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(device))[0]
        else:
            # Use dummy embeddings for now
            text_embeddings = torch.randn(1, 77, 768, device=device)
        
        # Prepare conditioning image
        condition_image = condition_image.to(device)
        
        # Initialize random noise
        shape = (1, self.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(shape, generator=generator, device=device)
        
        # Set timesteps
        if self.scheduler is not None:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
        else:
            # Simple linear schedule for demo
            timesteps = torch.linspace(1000, 0, num_inference_steps, device=device)
        
        # Denoising loop
        for t in timesteps:
            # Get ControlNet outputs
            latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            # Handle timestep properly - it should be a scalar or batch tensor
            if isinstance(t, torch.Tensor) and t.dim() == 0:
                t_input = t.expand(latent_input.shape[0])
            elif isinstance(t, torch.Tensor):
                t_input = t
            else:
                t_input = torch.tensor([t] * latent_input.shape[0], device=device)
            
            # Forward through ControlNet
            _, down_residuals, mid_residual = self.controlnet(
                latent_input,
                t_input,
                text_embeddings,
                condition_image,
                return_controlnet_outputs=True
            )
            
            # Apply conditioning scale
            down_residuals = [r * conditioning_scale for r in down_residuals]
            if mid_residual is not None:
                mid_residual = mid_residual * conditioning_scale
            
            # Standard U-Net forward with residuals
            noise_pred = self.unet(
                latent_input,
                t_input,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid_residual
            ).sample
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            if self.scheduler is not None:
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            else:
                # Simple Euler step
                latents = latents - 0.02 * noise_pred
        
        return latents