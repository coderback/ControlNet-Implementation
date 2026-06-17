# ControlNet Implementation

A from-scratch, **parity-verified** PyTorch implementation of ControlNet — *"Adding Conditional
Control to Text-to-Image Diffusion Models"* ([Zhang, Rao & Agrawala, 2023](https://arxiv.org/abs/2302.05543)) —
built on a frozen Stable Diffusion v1.5.

The ControlNet here is hand-written (not a wrapper around the library model). Its correctness is
**proven by a test** that asserts its outputs match Hugging Face `diffusers.ControlNetModel`
to ~1e-4 given identical weights — so the "I implemented it" claim is verifiable, not just asserted.

> **Status:** core architecture is being rewritten for a faithful, trainable implementation.
> See [`docs/`](docs/) and the plan for the current phase.

## How it works

ControlNet adds spatial control (Canny edges, depth, pose, …) to a frozen diffusion model:

1. **Freeze** the pretrained SD U-Net.
2. **Clone** its encoder + middle block into a trainable copy.
3. A small **conditioning network** maps the pixel-space condition image into the latent grid and
   is added to the trainable copy's input.
4. The copy's per-layer outputs pass through **zero convolutions** (1×1 convs initialised to zero,
   so training starts as an exact no-op) and are injected into the frozen U-Net's decoder as
   `down_block_additional_residuals` / `mid_block_additional_residual`.
5. Train with the standard DDPM noise-prediction loss and 50% prompt dropout; only the
   ControlNet learns.

## Repository layout

```
configs/        YAML training configs (local smoke / RunPod 24GB / A100)
data/           datasets (gitignored) — prepared by scripts/prepare_coco.py
docker/         Dockerfile + RunPod entrypoint
docs/           runpod.md ($8 runbook), image_requirements.md
scripts/        prepare_coco.py, train.py, evaluate.py, download_datasets.py
src/
  models/       zero_conv.py, conditioning.py, controlnet.py
  data/         dataset.py (COCO captions + on-the-fly Canny)
  training/     trainer.py (accelerate-based)
  inference/    generate.py
  evaluation/   metrics.py (FID, CLIP score, Canny fidelity)
tests/          test_parity_with_diffusers.py (headline), test_smoke_train.py, ...
notebooks/      demo.ipynb (inference/results visualization only)
```

## Quick start

```bash
pip install -e .[dev,train,eval]

# 1. Prepare data (COCO val2017 + captions; images may already be local)
python scripts/prepare_coco.py --root data/coco            # add --download on a fresh machine

# 2. Prove correctness for free (CPU, seconds)
pytest tests/

# 3. Smoke-test training locally (e.g. a 4GB laptop GPU)
accelerate launch scripts/train.py --config configs/smoke_local.yaml
```

## Training at scale (RunPod, ~$8 budget)

The full cost-disciplined workflow is in [`docs/runpod.md`](docs/runpod.md). Summary: the parity
test + local smoke run catch bugs for $0; the paid GPU is used only for scale, on a cheap 24GB
card, with the checkpoint downloaded before the pod is terminated.

```bash
accelerate launch scripts/train.py --config configs/runpod_24gb.yaml
```

## Evaluation

`scripts/evaluate.py` reports:
- **FID** — generated vs. real image distribution.
- **CLIP score** — text–image alignment.
- **Canny fidelity** — re-extract edges from the generated image and compare to the input
  condition (edge-F1 + SSIM): the ControlNet-specific "did it follow the control?" metric.

## Conditioning types

| Type | Input | Status |
|------|-------|--------|
| `canny` | grayscale edge map | primary |
| `depth`, `pose`, `seg`, `normal`, `scribble` | see `docs/image_requirements.md` | via `controlnet-aux`, extensible |

## Citation

```bibtex
@inproceedings{zhang2023controlnet,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  booktitle={ICCV},
  year={2023}
}
```

## License

MIT — see [LICENSE](LICENSE). Builds on Hugging Face Diffusers and Stable Diffusion (Stability AI).
