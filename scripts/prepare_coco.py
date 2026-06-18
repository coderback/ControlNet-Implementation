"""
Prepare a COCO-based ControlNet dataset (image + caption metadata; Canny generated on the fly).

The split decides the output file and role:
  --split val2017    -> val.jsonl    (held-out evaluation set)
  --split train2017  -> train.jsonl  (training set; cap with --max-images, e.g. 45000)

Only the images that have captions are kept, and for train2017 only the requested `--max-images`
are extracted from the 18GB zip (selective extraction — the rest is never unpacked).

Examples
--------
    # evaluation set
    python scripts/prepare_coco.py --split val2017 --download
    # training set: 45k from train2017
    python scripts/prepare_coco.py --split train2017 --max-images 45000 --download
"""

import argparse
import json
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

IMAGE_URL = "http://images.cocodataset.org/zips/{split}.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)


def ensure_captions(root: Path, split: str, download: bool) -> Path:
    captions = root / "annotations" / f"captions_{split}.json"
    if captions.exists():
        return captions
    if not download:
        raise FileNotFoundError(f"No captions at {captions}; re-run with --download.")
    zip_path = root / "annotations_trainval2017.zip"
    if not zip_path.exists():
        _download(ANNOTATIONS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root, members=[m for m in zf.namelist() if m.endswith(f"captions_{split}.json")])
    zip_path.unlink(missing_ok=True)
    return captions


def ensure_images(root: Path, split: str, needed: list, download: bool) -> None:
    img_dir = root / split
    missing = [f for f in needed if not (img_dir / f).exists()]
    if not missing:
        return
    if not download:
        raise FileNotFoundError(f"{len(missing)} {split} images missing in {img_dir}; re-run with --download.")
    zip_path = root / f"{split}.zip"
    if not zip_path.exists():
        _download(IMAGE_URL.format(split=split), zip_path)
    want = {f"{split}/{f}" for f in needed}
    print(f"Extracting {len(want)} images from {zip_path.name} (selective) ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root, members=[m for m in zf.namelist() if m in want])
    zip_path.unlink(missing_ok=True)  # keep the disk lean


def main():
    p = argparse.ArgumentParser(description="Prepare COCO ControlNet metadata")
    p.add_argument("--root", type=str, default="data/coco")
    p.add_argument("--split", type=str, default="val2017", choices=["val2017", "train2017"])
    p.add_argument("--max-images", type=int, default=None, help="Cap the number of images (train2017).")
    p.add_argument("--download", action="store_true", help="Fetch images/annotations if missing.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = Path(args.root)
    captions_path = ensure_captions(root, args.split, args.download)

    with open(captions_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    id_to_caption = {}
    for ann in coco["annotations"]:
        id_to_caption.setdefault(ann["image_id"], ann["caption"].strip())

    items = [(id_to_file[i], id_to_caption[i]) for i in id_to_file if i in id_to_caption]
    random.Random(args.seed).shuffle(items)
    if args.max_images:
        items = items[: args.max_images]

    ensure_images(root, args.split, [f for f, _ in items], args.download)
    rows = [{"image": f"{args.split}/{f}", "caption": c} for f, c in items if (root / args.split / f).exists()]

    out = root / ("train.jsonl" if args.split.startswith("train") else "val.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} samples ({args.split}) -> {out}")


if __name__ == "__main__":
    main()