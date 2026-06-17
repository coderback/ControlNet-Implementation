"""
Prepare a COCO-based ControlNet dataset.

Produces train/val metadata (JSONL: one {"image": <relative path>, "caption": <str>} per line)
from COCO val2017 images + the official caption annotations. Conditions (Canny edges) are
generated on the fly at training time, so nothing is precomputed here.

Typical use
-----------
Local (images already present at data/coco/val2017 from the earlier download):
    python scripts/prepare_coco.py --root data/coco

RunPod (nothing present yet — fetch images + annotations first):
    python scripts/prepare_coco.py --root data/coco --download

Outputs
-------
    <root>/train.jsonl
    <root>/val.jsonl
"""

import argparse
import json
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)


def ensure_images(root: Path, download: bool) -> Path:
    """Return the val2017 image directory, downloading + extracting if requested."""
    img_dir = root / "val2017"
    if img_dir.exists() and any(img_dir.glob("*.jpg")):
        return img_dir
    if not download:
        raise FileNotFoundError(
            f"No images at {img_dir}. Re-run with --download to fetch val2017."
        )
    zip_path = root / "val2017.zip"
    if not zip_path.exists():
        _download(VAL_IMAGES_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink(missing_ok=True)  # keep the disk lean
    return img_dir


def ensure_captions(root: Path, download: bool) -> Path:
    """Return path to captions_val2017.json, downloading the annotations if requested."""
    captions = root / "annotations" / "captions_val2017.json"
    if captions.exists():
        return captions
    if not download:
        raise FileNotFoundError(
            f"No captions at {captions}. Re-run with --download to fetch COCO annotations."
        )
    zip_path = root / "annotations_trainval2017.zip"
    if not zip_path.exists():
        _download(ANNOTATIONS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        # extract only the captions file we need
        members = [m for m in zf.namelist() if m.endswith("captions_val2017.json")]
        zf.extractall(root, members=members)
    zip_path.unlink(missing_ok=True)
    return captions


def build_metadata(img_dir: Path, captions_path: Path, root: Path, val_fraction: float, seed: int):
    with open(captions_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # image_id -> file_name
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    # image_id -> first caption (one caption per image is enough for ControlNet)
    id_to_caption: dict[int, str] = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in id_to_caption:
            id_to_caption[iid] = ann["caption"].strip()

    samples = []
    for iid, file_name in id_to_file.items():
        if iid not in id_to_caption:
            continue
        if not (img_dir / file_name).exists():
            continue
        samples.append({"image": f"val2017/{file_name}", "caption": id_to_caption[iid]})

    random.Random(seed).shuffle(samples)
    n_val = max(1, int(len(samples) * val_fraction))
    val, train = samples[:n_val], samples[n_val:]

    def write_jsonl(path: Path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "val.jsonl", val)
    print(f"Wrote {len(train)} train / {len(val)} val samples to {root}")


def main():
    p = argparse.ArgumentParser(description="Prepare COCO ControlNet metadata")
    p.add_argument("--root", type=str, default="data/coco", help="Dataset root")
    p.add_argument("--download", action="store_true", help="Fetch images/annotations if missing")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.root)
    img_dir = ensure_images(root, args.download)
    captions_path = ensure_captions(root, args.download)
    build_metadata(img_dir, captions_path, root, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
