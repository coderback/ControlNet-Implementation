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
        condition_features = self.condition_encoder(condition)
        
        # Standard U-Net forward pass with ControlNet modifications
        controlnet_residuals = []
        
        # Process through encoder blocks with ControlNet
        sample_down = sample
        for i, (down_block, controlnet_block, zero_conv) in enumerate(
            zip(self.unet.down_blocks, self.controlnet_down_blocks, self.zero_convs_down)
        ):
            # Run original down block (frozen)
            original_output = down_block(
                sample_down,
                temb=None,  # Simplified - would need proper time embedding
            )
            
            # Run ControlNet block (trainable)
            # Add condition features at the first level
            if i == 0:
                # Resize condition features to match current resolution
                h, w = sample_down.shape[-2:]
                condition_resized = torch.nn.functional.interpolate(
                    condition_features,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                )
                controlnet_input = sample_down + condition_resized
            else:
                controlnet_input = sample_down
                
            controlnet_output = controlnet_block(controlnet_input)
            
            # Apply zero convolution and add to residuals
            controlnet_residual = zero_conv(controlnet_output)
            controlnet_residuals.append(controlnet_residual * self.conditioning_scale)
            
            sample_down = original_output
        
        # Process middle block with ControlNet
        mid_residual = None
        if self.unet.mid_block is not None and self.controlnet_mid_block is not None:
            # Original middle block
            original_mid = self.unet.mid_block(sample_down)
            
            # ControlNet middle block
            controlnet_mid = self.controlnet_mid_block(sample_down)
            
            # Apply zero convolution
            mid_residual = self.zero_conv_mid(controlnet_mid) * self.conditioning_scale
            
            sample_down = original_mid
        
        if return_controlnet_outputs:
            return sample_down, controlnet_residuals, mid_residual
        
        # Continue with decoder (using controlnet residuals)
        return self._run_decoder_with_residuals(
            sample_down, controlnet_residuals, mid_residual,
            timestep, encoder_hidden_states
        )
    
    def _run_decoder_with_residuals(
        self,
        hidden_states: torch.Tensor,
        down_residuals: List[torch.Tensor],
        mid_residual: Optional[torch.Tensor],
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Run the decoder with ControlNet residuals added."""
        
        # Add middle residual
        if mid_residual is not None:
            hidden_states = hidden_states + mid_residual
        
        # Process through up blocks
        for i, up_block in enumerate(self.unet.up_blocks):
            # Get corresponding down residuals
            if i < len(down_residuals):
                # Add ControlNet residuals to skip connections
                res_samples = down_residuals[-(i+1):]  # Get residuals for this level
                
                # Run up block with modified residuals
                hidden_states = up_block(
                    hidden_states,
                    res_hidden_states_tuple=tuple(res_samples) if res_samples else None,
                    temb=None,  # Simplified
                )
            else:
                hidden_states = up_block(hidden_states)
        
        # Final output
        if hasattr(self.unet, 'conv_norm_out'):
            hidden_states = self.unet.conv_norm_out(hidden_states)
            hidden_states = self.unet.conv_act(hidden_states)
        
        if hasattr(self.unet, 'conv_out'):
            hidden_states = self.unet.conv_out(hidden_states)
        
        return hidden_states


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
            t_input = t.expand(latent_input.shape[0]) if hasattr(t, 'expand') else torch.tensor([t] * latent_input.shape[0], device=device)
            
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