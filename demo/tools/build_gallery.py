"""Slice the committed assets/diagnostic.png grid into per-sample panels for the gallery.

Every generated panel comes from the saved figure written by scripts/diagnose.py on the
12k-step checkpoint. No inference runs here. Captions are the first eight lines of
data/coco/val.jsonl, which is what diagnose.py iterated over; per-sample recall values are
the ones recorded in scripts/plot_training.py.

    python demo/tools/build_gallery.py --val data/coco/val.jsonl --captions-json <coco captions_val2017.json>
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OFF = [0.430, 0.266, 0.317, 0.282, 0.243, 0.295, 0.539, 0.143]
ON = [0.856, 0.791, 0.905, 0.898, 0.848, 0.779, 0.861, 0.887]


def spans(mask, min_len):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", required=True)
    p.add_argument("--captions-json", required=True)
    args = p.parse_args()

    im = Image.open(ROOT / "assets" / "diagnostic.png").convert("RGB")
    a = np.asarray(im).astype(int)
    nonwhite = (a.sum(axis=2) < 740)
    # Panel columns are the long runs of non-white pixels; titles only span part of a row.
    cols = spans(nonwhite.mean(axis=0) > 0.5, 150)
    rows = spans(nonwhite[:, cols[0][0]:cols[0][1]].mean(axis=1) > 0.9, 150)
    assert len(cols) == 4 and len(rows) == 8, (cols, rows)

    samples = [json.loads(l) for l in open(args.val, encoding="utf8")][:8]
    coco = json.load(open(args.captions_json, encoding="utf8"))
    licenses = {l["id"]: l for l in coco["licenses"]}
    images = {i["id"]: i for i in coco["images"]}

    out = ROOT / "demo" / "img"
    out.mkdir(parents=True, exist_ok=True)
    names = ["cond", "off", "on", "orig"]
    gallery = []
    for r, (y0, y1) in enumerate(rows):
        for c, (x0, x1) in enumerate(cols):
            im.crop((x0, y0, x1, y1)).save(out / f"s{r}-{names[c]}.webp", quality=88, method=6)
        coco_id = int(Path(samples[r]["image"]).stem)
        info = images[coco_id]
        gallery.append({
            "cocoId": coco_id,
            "prompt": " ".join(samples[r]["caption"].split()),
            "recallOff": OFF[r],
            "recallOn": ON[r],
            "license": licenses[info["license"]]["name"],
            "licenseUrl": licenses[info["license"]]["url"],
            "source": info["flickr_url"],
            "panel": [x1 - x0, y1 - y0],
        })
    (ROOT / "demo" / "data").mkdir(exist_ok=True)
    (ROOT / "demo" / "data" / "gallery.json").write_text(json.dumps(gallery, indent=1), encoding="utf8")
    print(cols, rows)


if __name__ == "__main__":
    main()
