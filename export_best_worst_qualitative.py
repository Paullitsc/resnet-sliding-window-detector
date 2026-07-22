#!/usr/bin/env python3
"""Build GT (green) + prediction (red) images for highest/lowest IoU rows in per_image_iou_results.csv."""
from __future__ import annotations

import re
import sys
from collections import namedtuple
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

Box = namedtuple("Box", ["x_min", "y_min", "x_max", "y_max"])
ROOT = Path(__file__).resolve().parent
IMAGES_ROOT = ROOT / "data/stanford_dogs/Images"
CSV_PATH = ROOT / "outputs/per_image_iou_results.csv"
OUT_DIR = ROOT / "outputs/best_worst"

BOX_RE = re.compile(
    r"Box\(x_min=(?P<x0>\d+),\s*y_min=(?P<y0>\d+),\s*x_max=(?P<x1>\d+),\s*y_max=(?P<y1>\d+)\)"
)


def parse_box(s: str) -> Optional[Box]:
    if not isinstance(s, str) or not s.strip():
        return None
    m = BOX_RE.search(s)
    if not m:
        return None
    g = m.groupdict()
    return Box(int(g["x0"]), int(g["y0"]), int(g["x1"]), int(g["y1"]))


def find_image(image_id: str) -> Optional[Path]:
    if not IMAGES_ROOT.is_dir():
        return None
    matches = list(IMAGES_ROOT.rglob(f"{image_id}.jpg"))
    return matches[0] if matches else None


def draw_overlay(
    bgr,
    pred: Box | None,
    gt: Box | None,
    iou: float,
    score: float,
) -> None:
    if gt is not None:
        cv2.rectangle(
            bgr, (gt.x_min, gt.y_min), (gt.x_max, gt.y_max), (0, 255, 0), 3
        )
    if pred is not None:
        cv2.rectangle(
            bgr, (pred.x_min, pred.y_min), (pred.x_max, pred.y_max), (0, 0, 255), 2
        )
    text = f"IoU: {iou:.3f}  Score: {score:.3f}"
    cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4)
    cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


def main() -> int:
    if not CSV_PATH.is_file():
        print(f"Missing {CSV_PATH}", file=sys.stderr)
        return 1
    df = pd.read_csv(CSV_PATH)
    df["iou"] = df["iou"].astype(float)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 5
    best = df.nlargest(n, "iou").reset_index(drop=True)
    worst = df.nsmallest(n, "iou").reset_index(drop=True)

    written = 0
    for tag, part in [("best", best), ("worst", worst)]:
        for rank, row in part.iterrows():
            img_path = find_image(row["image_id"])
            if img_path is None:
                print(f"Skip (image not found): {row['image_id']}")
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"Skip (cv2 read failed): {img_path}")
                continue
            pred = parse_box(row["pred_box"])
            gt = parse_box(row["gt_box"])
            score = float(row["pred_score"])
            iou = float(row["iou"])
            draw_overlay(bgr, pred, gt, iou, score)
            fname = f"{tag}_{rank + 1:02d}_iou{iou:.4f}_{row['image_id']}.jpg"
            out_path = OUT_DIR / fname
            cv2.imwrite(str(out_path), bgr)
            print(out_path.relative_to(ROOT))
            written += 1

    print(f"\nWrote {written} images to {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
