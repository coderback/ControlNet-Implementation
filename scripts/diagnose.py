"""
Convergence diagnostic for a ControlNet checkpoint. Answers "is the control actually working?"
two ways, which is how you detect ControlNet's "sudden convergence" during a long run:

  1. Zero-conv weight magnitude. The control path (controlnet_down_blocks / mid / cond-embed
     conv_out) starts at EXACTLY 0 and grows as control is learned. mean|w| ~5e-4 is essentially
     unconverged; it needs to climb toward the ~1e-2 scale of a normal trained conv.
  2. Control OFF vs ON. Generates each prompt twice from the same seed (scale 0 vs 1) and reports
     `edge_recall` (fraction of the conditioning Canny edges reproduced). If ON >> OFF and the ON
     column visibly tracks the edges, control has emerged. Saves a grid to --output.

Usage:
    python scripts/diagnose.py --controlnet outputs/full/controlnet-8000.safetensors --config configs/full.yaml
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

from src.evaluation.metrics import canny_fidelity
from src.inference.generate import ControlNetInference


def zero_conv_report(ckpt_path: str) -> None:
    sd = load_file(ckpt_path)

    def stats(substr):
        ts = [v for k, v in sd.items() if substr in k and k.endswith("weight")]
        if not ts:
            return None
        w = torch.cat([t.flatten().float() for t in ts])
        return w.abs().mean().item(), w.abs().max().item()

    print("=== zero-init control path (started at 0; mean|w| ~1e-2 == converged, ~5e-4 == not) ===")
    for label, key in [("down zero-convs", "controlnet_down_blocks"),
                       ("mid  zero-conv ", "controlnet_mid_block"),
                       ("cond-embed out ", "controlnet_cond_embedding.conv_out")]:
        s = stats(key)
        if s:
            print(f"  {label}: mean|w|={s[0]:.2e}  max|w|={s[1]:.3f}")
    ref = stats("down_blocks.0.resnets.0.conv1")
    if ref:
        print(f"  (reference normal conv:   mean|w|={ref[0]:.2e}  max|w|={ref[1]:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controlnet", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--output", default="outputs/diagnostic.png")
    args = p.parse_args()

    zero_conv_report(args.controlnet)

    cfg = OmegaConf.load(args.config)
    root = Path(cfg.data.root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    infer = ControlNetInference(model_id=cfg.model.base_model_id, controlnet_path=args.controlnet,
                                condition_type=cfg.model.condition_type, device=device)
    infer.pipe.enable_attention_slicing(); infer.pipe.enable_vae_slicing()

    def gen(cap, img, scale):
        for res in [cfg.data.image_size, 384, 256]:
            try:
                return infer.generate(cap, Image.fromarray(img), num_inference_steps=25,
                                      image_resolution=res, controlnet_conditioning_scale=scale, seed=0)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

    samples = [json.loads(l) for l in open(root / "val.jsonl")][: args.n]
    fig, ax = plt.subplots(args.n, 4, figsize=(16, 4 * args.n))
    on_rec, off_rec = [], []
    print("\n=== control OFF vs ON (edge_recall: fraction of conditioning edges reproduced) ===")
    for i, s in enumerate(samples):
        img = np.array(Image.open(root / s["image"]).convert("RGB").resize((512, 512)))
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cfg.data.canny_low, cfg.data.canny_high)
        cond3 = np.stack([edges] * 3, -1)
        off, on = gen(s["caption"], img, 0.0), gen(s["caption"], img, 1.0)
        r_off = canny_fidelity(off, cond3, cfg.data.canny_low, cfg.data.canny_high)["edge_recall"]
        r_on = canny_fidelity(on, cond3, cfg.data.canny_low, cfg.data.canny_high)["edge_recall"]
        off_rec.append(r_off); on_rec.append(r_on)
        for j, im in enumerate([edges, off, on, img]):
            ax[i, j].imshow(im, cmap="gray" if j == 0 else None); ax[i, j].axis("off")
        ax[i, 1].set_title(f"OFF recall={r_off:.2f}"); ax[i, 2].set_title(f"ON recall={r_on:.2f}")
        print(f"  sample {i}: OFF={r_off:.3f}  ON={r_on:.3f}  | {s['caption'][:40]}")
    gap = float(np.mean(on_rec) - np.mean(off_rec))
    print(f"\n  MEAN edge_recall  ON={np.mean(on_rec):.3f}  OFF={np.mean(off_rec):.3f}  gap={gap:+.3f}")
    print("  -> a clear positive gap (and ON visibly tracking edges) means control has emerged.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.output, dpi=70)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
