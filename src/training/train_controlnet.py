"""
Training script for ControlNet.

This implements the training loop for ControlNet following the paper's methodology:
- Locks the original U-Net parameters
- Trains only the ControlNet (trainable copy) parameters
- Uses the same loss function as the original diffusion model
- Implements the "sudden convergence phenomenon"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import wandb
from pathlib import Path
import json

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

from ..models.controlnet import ControlNet
from ..data.dataset import ControlNetDataset


class ControlNetTrainer:
    """
    Trainer class for ControlNet.
    
    Implements the training procedure described in the ControlNet paper,
    including the zero convolution initialization and gradual learning.
    """
    
    def __init__(
        self,
        controlnet: ControlNet,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        noise_scheduler: DDPMScheduler,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        num_train_epochs: int = 100,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_every: int = 1000,
        log_every: int = 100,
        output_dir: str = "./outputs",
        resume_from_checkpoint: Optional[str] = None,
        use_wandb: bool = True
    ):
        self.controlnet = controlnet
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        self.save_every = save_every
        self.log_every = log_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        
        # Setup training components
        self._setup_optimizer()
        self._setup_scheduler()
        
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Load checkpoint if provided
        self.global_step = 0
        self.epoch = 0
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        # Move models to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._move_to_device()
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project="controlnet-training",
                config={
                    "learning_rate": learning_rate,
                    "num_epochs": num_train_epochs,
                    "batch_size": train_dataloader.batch_size,
                    "condition_type": controlnet.condition_type
                }
            )
    
    def _setup_optimizer(self):
        """Setup optimizer for ControlNet parameters only."""
        # Only train ControlNet parameters, not the original U-Net
        trainable_params = list(self.controlnet.parameters())
        
        # Remove frozen U-Net parameters
        trainable_params = [p for p in trainable_params if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        print(f"Training {sum(p.numel() for p in trainable_params):,} parameters")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        num_training_steps = len(self.train_dataloader) * self.num_train_epochs
        
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps
        )
    
    def _move_to_device(self):
        """Move models to training device."""
        self.controlnet.to(self.device)
        self.unet.to(self.device)
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        
        # Ensure U-Net is in eval mode (frozen)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        
        # ControlNet in training mode
        self.controlnet.train()
    
    def _encode_prompt(self, prompts: list) -> torch.Tensor:
        """Encode text prompts to embeddings."""
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        
        return text_embeddings
    
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space."""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the diffusion loss with ControlNet conditioning.
        
        This implements Equation (5) from the paper:
        L = E[||ε - ε_θ(z_t, t, c_t, c_f)||_2^2]
        
        Where:
        - ε is the noise added to the image
        - ε_θ is the predicted noise
        - z_t is the noisy latent at timestep t
        - c_t is the text conditioning
        - c_f is the ControlNet conditioning
        """
        # Get batch data
        images = batch["image"].to(self.device)
        conditions = batch["condition"].to(self.device)
        prompts = batch["prompt"]
        
        # Encode inputs
        latents = self._encode_images(images)
        text_embeddings = self._encode_prompt(prompts)
        
        # Sample random timesteps
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get ControlNet residuals
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            _, down_residuals, mid_residual = self.controlnet(
                noisy_latents,
                timesteps,
                text_embeddings,
                conditions,
                return_controlnet_outputs=True
            )
            
            # Predict noise with ControlNet conditioning
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid_residual
            ).sample
            
            # Compute loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        # Log metrics
        metrics = {
            "loss": loss.item(),
            "learning_rate": self.lr_scheduler.get_last_lr()[0]
        }
        
        return loss, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        
        # Compute loss
        loss, metrics = self.compute_loss(batch)
        
        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.controlnet.parameters(),
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.controlnet.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
            
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataloader is None:
            return {}
        
        self.controlnet.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Validating"):
            loss, _ = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        
        self.controlnet.train()
        
        return {"val_loss": total_loss / num_batches}
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint."""
        checkpoint = {
            "controlnet_state_dict": self.controlnet.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_step": step,
            "epoch": self.epoch,
        }
        
        if self.mixed_precision:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        checkpoint_path = self.output_dir / f"checkpoint-{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save just the ControlNet model
        controlnet_path = self.output_dir / f"controlnet-{step}.pt"
        torch.save(self.controlnet.state_dict(), controlnet_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        self.controlnet.load_state_dict(checkpoint["controlnet_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        
        if self.mixed_precision and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.num_train_epochs} epochs")
        print(f"Training on {self.device}")
        print(f"Total training steps: {len(self.train_dataloader) * self.num_train_epochs}")
        
        # Implement prompt dropout (50% as mentioned in paper)
        for epoch in range(self.epoch, self.num_train_epochs):
            self.epoch = epoch
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.num_train_epochs}"
            )
            
            for batch in progress_bar:
                # Random prompt dropout (50% of the time)
                if np.random.random() < 0.5:
                    batch["prompt"] = [""] * len(batch["prompt"])
                
                # Training step
                metrics = self.train_step(batch)
                
                # Update progress bar
                progress_bar.set_postfix(metrics)
                
                # Logging
                if self.global_step % self.log_every == 0:
                    if self.use_wandb:
                        wandb.log(metrics, step=self.global_step)
                
                # Validation
                if self.global_step % (self.log_every * 5) == 0:
                    val_metrics = self.validate()
                    if val_metrics and self.use_wandb:
                        wandb.log(val_metrics, step=self.global_step)
                
                # Save checkpoint
                if self.global_step % self.save_every == 0 and self.global_step > 0:
                    self.save_checkpoint(self.global_step)
                
                self.global_step += 1
        
        # Final save
        self.save_checkpoint(self.global_step)
        print("Training completed!")


def create_trainer_from_config(config_path: str) -> ControlNetTrainer:
    """Create trainer from configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load models based on config
    # This would load the actual Stable Diffusion components
    # Implementation depends on your specific setup
    
    return ControlNetTrainer(**config)