"""
Evaluation metrics for a trained ControlNet.

  - FID            : Frechet Inception Distance (generated vs real) via torchmetrics.
  - CLIP score     : text-image alignment via torchmetrics.
  - Canny fidelity : re-extract Canny from the generated image and compare to the INPUT condition
                     (edge-F1 + SSIM). The ControlNet-specific "did it follow the control?" metric.
"""

from typing import Sequence

import numpy as np
import torch
from PIL import Image


def _to_uint8_batch(images: Sequence[Image.Image]) -> torch.Tensor:
    arr = np.stack([np.array(im.convert("RGB")) for im in images])  # [N,H,W,3]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()    # uint8 [N,3,H,W]


def compute_fid(generated: Sequence[Image.Image], real: Sequence[Image.Image]) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False)
    fid.update(_to_uint8_batch(real), real=True)
    fid.update(_to_uint8_batch(generated), real=False)
    return float(fid.compute())


def compute_clip_score(images: Sequence[Image.Image], prompts: Sequence[str]) -> float:
    from torchmetrics.multimodal.clip_score import CLIPScore
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    metric.update(_to_uint8_batch(images), list(prompts))
    return float(metric.compute())


def canny_fidelity(generated: Image.Image, condition: np.ndarray,
                   low: int = 100, high: int = 200, tolerance: int = 2) -> dict:
    """Re-extract Canny from `generated`, compare to the input `condition` edge map.

    Edges almost never align pixel-exactly, so matching is done with a `tolerance` (each edge map is
    dilated before overlap). Returns:
      - edge_recall:    fraction of the *conditioning* edges reproduced in the generation (the key
                        "did it follow the control?" number) — within `tolerance` px.
      - edge_f1:        F1 of the tolerant precision/recall.
      - ssim:           SSIM of the raw edge maps (kept for continuity; insensitive — see README).
    """
    import cv2
    from skimage.metrics import structural_similarity as ssim

    gen = np.array(generated.convert("RGB"))
    gen_edges = cv2.Canny(cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY), low, high)

    cond = condition[..., 0] if condition.ndim == 3 else condition
    cond = cv2.resize(cond, (gen_edges.shape[1], gen_edges.shape[0]), interpolation=cv2.INTER_NEAREST)

    g = gen_edges > 0
    c = cond > 0
    k = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    g_dil = cv2.dilate(g.astype(np.uint8), k) > 0
    c_dil = cv2.dilate(c.astype(np.uint8), k) > 0

    precision = float(np.logical_and(g, c_dil).sum() / (g.sum() + 1e-8))   # gen edges near a cond edge
    recall = float(np.logical_and(c, g_dil).sum() / (c.sum() + 1e-8))      # cond edges reproduced
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    ssim_val = ssim(gen_edges, cond, data_range=255)
    return {"edge_f1": float(f1), "edge_recall": recall, "ssim": float(ssim_val)}
