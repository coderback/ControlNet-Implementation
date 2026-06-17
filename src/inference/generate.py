"""
Inference for ControlNet.

Loads a base Stable Diffusion pipeline, attaches a trained `ControlNet`, and runs a DDIM sampling
loop with classifier-free guidance, injecting the ControlNet residuals into the frozen U-Net at
every step. Condition images are normalised to [0, 1] to match training (see CocoCannyDataset).
"""

from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import DDIMScheduler, StableDiffusionPipeline

from ..data.preprocess import preprocess_canny
from ..models.controlnet import ControlNet


class ControlNetInference:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_path: Optional[str] = None,
        condition_type: str = "canny",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        self.condition_type = condition_type

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.controlnet = ControlNet.from_unet(self.pipe.unet, copy_weights=False)
        if controlnet_path:
            self.load_controlnet_weights(controlnet_path)

        self.pipe = self.pipe.to(device)
        self.controlnet = self.controlnet.to(device, dtype=dtype)
        print(f"Initialized ControlNet inference for {condition_type} conditioning")

    def load_controlnet_weights(self, checkpoint_path: str):
        path = Path(checkpoint_path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(path))
        else:
            state_dict = torch.load(str(path), map_location="cpu")
            state_dict = state_dict.get("controlnet_state_dict", state_dict)
        missing, unexpected = self.controlnet.load_state_dict(state_dict, strict=False)
        print(f"Loaded {checkpoint_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    def preprocess_condition(self, condition_input: Union[str, np.ndarray, Image.Image],
                             resolution: int = 512) -> torch.Tensor:
        if isinstance(condition_input, str):
            condition_input = np.array(Image.open(condition_input).convert("RGB"))
        elif isinstance(condition_input, Image.Image):
            condition_input = np.array(condition_input.convert("RGB"))

        if self.condition_type == "canny":
            edges = preprocess_canny(condition_input)          # [H, W, 1]
            condition = np.repeat(edges, 3, axis=-1)
        else:
            condition = condition_input

        condition = cv2.resize(condition, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(condition).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # [1,3,H,W] in [0,1]
        return tensor.to(self.device, dtype=self.dtype)

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
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        condition = self.preprocess_condition(condition_input, image_resolution)

        # text embeddings (uncond, cond) for CFG
        tok = self.pipe.tokenizer([negative_prompt, prompt], padding="max_length",
                                  max_length=self.pipe.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt").to(self.device)
        text_embeddings = self.pipe.text_encoder(tok.input_ids)[0]

        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, image_resolution // 8, image_resolution // 8),
            generator=generator, device=self.device, dtype=self.dtype,
        )
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        do_cfg = guidance_scale > 1.0
        for t in self.pipe.scheduler.timesteps:
            latent_in = torch.cat([latents] * 2) if do_cfg else latents
            latent_in = self.pipe.scheduler.scale_model_input(latent_in, t)
            cond_in = torch.cat([condition] * 2) if do_cfg else condition

            down, mid = self.controlnet(
                latent_in, t, text_embeddings, controlnet_cond=cond_in,
                conditioning_scale=controlnet_conditioning_scale, return_dict=False,
            )
            noise_pred = self.pipe.unet(
                latent_in, t, encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down, mid_block_additional_residual=mid,
            ).sample

            if do_cfg:
                uncond, cond_pred = noise_pred.chunk(2)
                noise_pred = uncond + guidance_scale * (cond_pred - uncond)

            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()[0]
        return Image.fromarray((image * 255).round().astype(np.uint8))

    def generate_batch(self, prompts: List[str], conditions: List, **kwargs) -> List[Image.Image]:
        return [self.generate(p, c, **kwargs) for p, c in zip(prompts, conditions)]
