"""
Example training script for ControlNet.

This demonstrates how to train a ControlNet model from scratch using your own dataset.
"""

import torch
from pathlib import Path
import argparse

from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.controlnet import ControlNet
from src.training.train_controlnet import ControlNetTrainer
from src.data.dataset import create_dataset, create_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train ControlNet")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of training data")
    parser.add_argument("--condition-type", type=str, default="canny", help="Type of conditioning")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model ID")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--resume-from", type=str, help="Resume training from checkpoint")
    parser.add_argument("--max-samples", type=int, help="Maximum number of training samples")
    parser.add_argument("--image-size", type=int, default=512, help="Training image size")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=100, help="Log metrics every N steps")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    print("Loading Stable Diffusion components...")
    
    # Load Stable Diffusion components
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder="unet",
        torch_dtype=torch.float32
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float32
    )
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id,
        subfolder="tokenizer"
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model_id,
        subfolder="scheduler"
    )
    
    print("Creating ControlNet...")
    
    # Create ControlNet
    controlnet = ControlNet(
        unet=unet,
        condition_type=args.condition_type
    )
    
    print("Loading datasets...")
    
    # Create datasets
    train_dataset = create_dataset(
        data_root=args.data_root,
        condition_type=args.condition_type,
        image_size=args.image_size,
        split="train",
        max_samples=args.max_samples
    )
    
    val_dataset = create_dataset(
        data_root=args.data_root,
        condition_type=args.condition_type,
        image_size=args.image_size,
        split="val",
        max_samples=min(100, args.max_samples) if args.max_samples else 100
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print("Setting up trainer...")
    
    # Create trainer
    trainer = ControlNetTrainer(
        controlnet=controlnet,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        mixed_precision=args.mixed_precision,
        save_every=args.save_every,
        log_every=args.log_every,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from,
        use_wandb=not args.no_wandb
    )
    
    print("Starting training...")
    
    # Start training
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()