"""
ControlNet training loop built on Hugging Face `accelerate`.

`accelerate` owns mixed precision (fp16/bf16), device placement, and gradient accumulation, so this
module only handles the ControlNet-specific logic, following the diffusers `train_controlnet.py`
reference recipe:

  - freeze VAE / text encoder / U-Net; train only the ControlNet (initialised from the U-Net)
  - VAE-encode image -> latent; add noise at a random timestep (DDPM)
  - ControlNet residuals -> frozen U-Net -> MSE noise loss (paper Eq. 5)
  - 50% prompt dropout, gradient clipping, configurable LR schedule, periodic safetensors checkpoints

Validation-sample logging is intentionally left to the walkthrough notebook / scripts/evaluate.py,
which load a saved checkpoint — keeping the training loop lean for 4GB GPUs. Run via:

    accelerate launch scripts/train.py --config configs/smoke_local.yaml
"""

import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer

from ..data.dataset import CocoCannyDataset, create_dataloader
from ..models.controlnet import ControlNet


def _enable_gradient_checkpointing(controlnet: ControlNet) -> None:
    """Enable gradient checkpointing on the ControlNet's diffusers blocks.

    Replicates diffusers' `ModelMixin.enable_gradient_checkpointing`: each block calls
    `self._gradient_checkpointing_func(...)`, so setting only the boolean flag is not enough —
    the checkpoint function must be assigned too. (Our ControlNet is a plain nn.Module, so we do
    this by hand rather than inheriting the ModelMixin machinery.)
    """
    def gc_func(module, *args):
        return torch.utils.checkpoint.checkpoint(module.__call__, *args, use_reentrant=False)

    for module in controlnet.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True
            module._gradient_checkpointing_func = gc_func


def train(config: DictConfig) -> None:
    m, d, tr, lg = config.model, config.data, config.training, config.logging

    accelerator = Accelerator(
        gradient_accumulation_steps=tr.gradient_accumulation_steps,
        mixed_precision=tr.mixed_precision,
        log_with="wandb" if lg.report_to == "wandb" else None,
    )
    if tr.get("seed") is not None:
        set_seed(tr.seed)

    output_dir = Path(tr.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load frozen components ---
    tokenizer = CLIPTokenizer.from_pretrained(m.base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(m.base_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(m.base_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(m.base_model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(m.base_model_id, subfolder="scheduler")

    controlnet = ControlNet.from_unet(unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()
    if tr.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        _enable_gradient_checkpointing(controlnet)

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=tr.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

    train_ds = CocoCannyDataset(d.root, "train", d.image_size, d.canny_low, d.canny_high,
                                d.get("max_train_samples"))
    train_dl = create_dataloader(train_ds, tr.train_batch_size, shuffle=True)

    lr_scheduler = get_scheduler(
        tr.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=tr.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=tr.max_train_steps * accelerator.num_processes,
    )

    controlnet, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dl, lr_scheduler
    )

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(tr.mixed_precision, torch.float32)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process and lg.report_to == "wandb":
        accelerator.init_trackers("controlnet", config=dict(
            condition=m.condition_type, image_size=d.image_size,
            lr=tr.learning_rate, batch=tr.train_batch_size,
        ))

    def save_controlnet(tag: str):
        if not accelerator.is_main_process:
            return
        state = {k: v.detach().cpu() for k, v in accelerator.unwrap_model(controlnet).state_dict().items()}
        path = output_dir / f"controlnet-{tag}.safetensors"
        save_file(state, str(path))
        print(f"Saved {path}")

    global_step = 0
    # `global_step` counts optimizer steps (one per `gradient_accumulation_steps` micro-batches),
    # so derive epochs from optimizer-steps-per-epoch, not raw dataloader length, or the loop ends
    # early (e.g. 50-image smoke set: 50 batches / accum 4 = 12 steps/epoch).
    steps_per_epoch = max(1, len(train_dl) // tr.gradient_accumulation_steps)
    num_epochs = math.ceil(tr.max_train_steps / steps_per_epoch)
    print(f"Training {sum(p.numel() for p in controlnet.parameters() if p.requires_grad):,} params "
          f"for {tr.max_train_steps} steps")

    for _ in range(num_epochs):
        for batch in train_dl:
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 50% prompt dropout (paper)
                captions = ["" if random.random() < tr.prompt_dropout else c for c in batch["captions"]]
                input_ids = tokenizer(captions, padding="max_length", truncation=True,
                                      max_length=tokenizer.model_max_length,
                                      return_tensors="pt").input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                cond = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                down, mid = controlnet(noisy_latents, timesteps, encoder_hidden_states,
                                       controlnet_cond=cond, return_dict=False)
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states,
                    down_block_additional_residuals=[r.to(dtype=weight_dtype) for r in down],
                    mid_block_additional_residual=mid.to(dtype=weight_dtype),
                ).sample

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), tr.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    accelerator.log(logs, step=global_step)
                    if global_step % 25 == 0:
                        print(f"step {global_step}/{tr.max_train_steps}  loss {logs['loss']:.4f}")
                if global_step % tr.checkpointing_steps == 0:
                    save_controlnet(str(global_step))
                if global_step >= tr.max_train_steps:
                    break
        if global_step >= tr.max_train_steps:
            break

    save_controlnet("final")
    accelerator.end_training()
    print("Training complete.")
