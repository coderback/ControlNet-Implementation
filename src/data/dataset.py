"""
Datasets for ControlNet training.

`CocoCannyDataset` is the primary dataset: it reads the JSONL metadata produced by
`scripts/prepare_coco.py` (one {"image": <relative path>, "caption": <str>} per line), loads each
image, resizes to a square `image_size`, and generates the Canny edge condition on the fly. No
precomputed condition files are needed.

Returns per sample:
    pixel_values            : float32 [3, H, W] in [-1, 1]  (target image for the VAE)
    conditioning_pixel_values: float32 [3, H, W] in [0, 1]  (Canny edges, fed to ControlNet)
    caption                 : str
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocess import make_canny


class CocoCannyDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 512,
        canny_low: int = 100,
        canny_high: int = 200,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.canny_low = canny_low
        self.canny_high = canny_high

        metadata = self.root / f"{split}.jsonl"
        if not metadata.exists():
            raise FileNotFoundError(
                f"{metadata} not found — run `python scripts/prepare_coco.py --root {root}` first."
            )
        with open(metadata, "r", encoding="utf-8") as f:
            self.samples: List[Dict] = [json.loads(line) for line in f if line.strip()]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} {split} samples from {metadata}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, rel_path: str) -> np.ndarray:
        image = Image.open(self.root / rel_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )
        return np.array(image)  # [H, W, 3] uint8

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        condition = make_canny(image, self.canny_low, self.canny_high)

        # target image -> [-1, 1]
        pixel_values = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1.0
        # condition -> [0, 1]
        conditioning_pixel_values = torch.from_numpy(condition).float().permute(2, 0, 1) / 255.0

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "caption": sample.get("caption", ""),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "conditioning_pixel_values": torch.stack([b["conditioning_pixel_values"] for b in batch]),
        "captions": [b["caption"] for b in batch],
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
