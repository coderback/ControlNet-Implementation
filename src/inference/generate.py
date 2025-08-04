"""
Inference script for ControlNet.

This script provides functionality to generate images using trained ControlNet models
with various types of conditioning inputs.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from typing import Optional, Union, List
from pathlib import Path
import argparse

from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

from ..models.controlnet import ControlNet, ControlNetPipeline
from ..models.condition_encoder import preprocess_canny, preprocess_depth


class ControlNetInference:
    """
    Inference class for ControlNet image generation.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_path: Optional[str] = None,
        condition_type: str = "canny",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize ControlNet inference pipeline.
        
        Args:
            model_id: Hugging Face model ID for base Stable Diffusion
            controlnet_path: Path to trained ControlNet weights
            condition_type: Type of conditioning ('canny', 'depth', etc.)
            device: Device to run inference on
            dtype: Data type for inference
        """
        self.device = device
        self.dtype = dtype
        self.condition_type = condition_type
        
        # Load Stable Diffusion components
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Create ControlNet
        self.controlnet = ControlNet(
            unet=self.pipe.unet,
            condition_type=condition_type
        )
        
        # Load ControlNet weights if provided
        if controlnet_path:
            self.load_controlnet_weights(controlnet_path)
        
        # Move to device
        self.pipe = self.pipe.to(device)
        self.controlnet = self.controlnet.to(device)
        
        # Use DDIM scheduler for faster inference
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        print(f"Initialized ControlNet inference for {condition_type} conditioning")
    
    def load_controlnet_weights(self, checkpoint_path: str):
        """Load ControlNet weights from checkpoint."""
        if Path(checkpoint_path).suffix == '.pt':
            # PyTorch checkpoint
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'controlnet_state_dict' in state_dict:
                state_dict = state_dict['controlnet_state_dict']
        else:
            # Assume it's a state dict directly
            state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        self.controlnet.load_state_dict(state_dict, strict=False)
        print(f"Loaded ControlNet weights from {checkpoint_path}")
    
    def preprocess_condition(
        self,
        condition_input: Union[str, np.ndarray, Image.Image],
        image_resolution: int = 512
    ) -> torch.Tensor:
        """
        Preprocess conditioning input based on condition type.
        
        Args:
            condition_input: Path to image, numpy array, or PIL Image
            image_resolution: Target resolution
        
        Returns:
            Preprocessed condition tensor
        """
        # Load image if path provided
        if isinstance(condition_input, str):
            if self.condition_type == "canny" and not Path(condition_input).exists():
                # Generate Canny from regular image
                image = np.array(Image.open(condition_input).convert("RGB"))
                condition = preprocess_canny(image)
            else:
                condition = np.array(Image.open(condition_input))
        elif isinstance(condition_input, Image.Image):
            condition = np.array(condition_input)
        else:
            condition = condition_input
        
        # Apply condition-specific preprocessing
        if self.condition_type == "canny":
            if len(condition.shape) == 3 and condition.shape[2] == 3:
                # Convert RGB image to Canny edges
                condition = preprocess_canny(condition)
            elif len(condition.shape) == 3 and condition.shape[2] == 1:
                # Already processed Canny edges
                pass
            elif len(condition.shape) == 2:
                # Grayscale Canny edges
                condition = condition[..., None]
        
        elif self.condition_type == "depth":
            if len(condition.shape) == 2:
                condition = preprocess_depth(condition)
            elif len(condition.shape) == 3 and condition.shape[2] == 1:
                condition = preprocess_depth(condition.squeeze(-1))
        
        # Resize to target resolution
        if condition.shape[:2] != (image_resolution, image_resolution):
            condition = cv2.resize(
                condition,
                (image_resolution, image_resolution),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Ensure 3 channels for consistency
        if len(condition.shape) == 2:
            condition = np.stack([condition] * 3, axis=-1)
        elif condition.shape[2] == 1:
            condition = np.repeat(condition, 3, axis=-1)
        
        # Convert to tensor and normalize
        condition = torch.from_numpy(condition).float() / 255.0
        condition = condition.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        condition = condition.to(self.device, dtype=self.dtype)
        
        # Normalize to [-1, 1] range
        condition = condition * 2.0 - 1.0
        
        return condition
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        condition_input: Union[str, np.ndarray, Image.Image],
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        image_resolution: int = 512,
        seed: Optional[int] = None,
        eta: float = 0.0
    ) -> Image.Image:
        """
        Generate image with ControlNet conditioning.
        
        Args:
            prompt: Text prompt
            condition_input: Conditioning input (image path, array, or PIL Image)
            negative_prompt: Negative text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            controlnet_conditioning_scale: ControlNet conditioning strength
            image_resolution: Output image resolution
            seed: Random seed for reproducibility
            eta: DDIM eta parameter
        
        Returns:
            Generated PIL Image
        """
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Preprocess condition
        condition = self.preprocess_condition(condition_input, image_resolution)
        
        # Encode prompts
        text_inputs = self.pipe.tokenizer(
            [prompt, negative_prompt],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        text_embeddings = self.pipe.text_encoder(text_inputs.input_ids)[0]
        
        # Split embeddings for CFG
        if guidance_scale > 1.0:
            text_embeddings_cond, text_embeddings_uncond = text_embeddings.chunk(2)
            text_embeddings = torch.cat([text_embeddings_uncond, text_embeddings_cond])
        
        # Initialize latents
        latents_shape = (1, self.pipe.unet.config.in_channels, image_resolution // 8, image_resolution // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=self.dtype)
        
        # Set scheduler timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Get ControlNet residuals
            _, down_residuals, mid_residual = self.controlnet(
                latent_model_input,
                t,
                text_embeddings,
                condition,
                return_controlnet_outputs=True
            )
            
            # Scale ControlNet outputs
            down_residuals = [residual * controlnet_conditioning_scale for residual in down_residuals]
            if mid_residual is not None:
                mid_residual = mid_residual * controlnet_conditioning_scale
            
            # Predict noise with ControlNet conditioning
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid_residual
            ).sample
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.pipe.scheduler.step(
                noise_pred, t, latents, eta=eta, use_clipped_model_output=False
            ).prev_sample
        
        # Decode latents to image
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        images = self.pipe.vae.decode(latents).sample
        
        # Post-process image
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(images[0])
        
        return image
    
    def generate_batch(
        self,
        prompts: List[str],
        condition_inputs: List[Union[str, np.ndarray, Image.Image]],
        **kwargs
    ) -> List[Image.Image]:
        """Generate multiple images in batch."""
        results = []
        
        for prompt, condition_input in zip(prompts, condition_inputs):
            image = self.generate(prompt, condition_input, **kwargs)
            results.append(image)
        
        return results


def main():
    """Command line interface for ControlNet inference."""
    parser = argparse.ArgumentParser(description="Generate images with ControlNet")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--condition", type=str, required=True, help="Path to conditioning image")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--controlnet-path", type=str, help="Path to ControlNet weights")
    parser.add_argument("--condition-type", type=str, default="canny", choices=["canny", "depth", "pose"])
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--controlnet-scale", type=float, default=1.0, help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = ControlNetInference(
        controlnet_path=args.controlnet_path,
        condition_type=args.condition_type
    )
    
    # Generate image
    image = inference.generate(
        prompt=args.prompt,
        condition_input=args.condition,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        image_resolution=args.resolution,
        seed=args.seed
    )
    
    # Save result
    image.save(args.output)
    print(f"Generated image saved to {args.output}")


if __name__ == "__main__":
    main()