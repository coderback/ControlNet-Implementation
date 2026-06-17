"""
Evaluation entrypoint.

    python scripts/evaluate.py --controlnet outputs/runpod_24gb/controlnet-final.safetensors \
        --config configs/runpod_24gb.yaml --num-samples 200

Generates images for the val split conditions, then reports FID / CLIP score / Canny fidelity
and writes a JSON report.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data.dataset import CocoCannyDataset
from src.evaluation.metrics import canny_fidelity, compute_clip_score, compute_fid
from src.inference.generate import ControlNetInference


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ControlNet")
    parser.add_argument("--controlnet", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="outputs/eval_report.json")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CocoCannyDataset(cfg.data.root, "val", cfg.data.image_size,
                          cfg.data.canny_low, cfg.data.canny_high, args.num_samples)
    infer = ControlNetInference(
        model_id=cfg.model.base_model_id, controlnet_path=args.controlnet,
        condition_type=cfg.model.condition_type, device=device,
    )

    generated, reals, prompts, fidelities = [], [], [], []
    for i in range(len(ds)):
        sample = ds[i]
        real = ((sample["pixel_values"] + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        cond = (sample["conditioning_pixel_values"] * 255).byte().permute(1, 2, 0).numpy()
        prompt = sample["caption"]

        gen = infer.generate(prompt, Image.fromarray(cond), image_resolution=cfg.data.image_size,
                             controlnet_conditioning_scale=cfg.model.conditioning_scale, seed=i)
        generated.append(gen)
        reals.append(Image.fromarray(real))
        prompts.append(prompt)
        fidelities.append(canny_fidelity(gen, cond, cfg.data.canny_low, cfg.data.canny_high))
        if (i + 1) % 25 == 0:
            print(f"generated {i + 1}/{len(ds)}")

    report = {
        "num_samples": len(generated),
        "fid": compute_fid(generated, reals),
        "clip_score": compute_clip_score(generated, prompts),
        "canny_edge_f1": float(np.mean([f["edge_f1"] for f in fidelities])),
        "canny_ssim": float(np.mean([f["ssim"] for f in fidelities])),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
