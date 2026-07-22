#!/usr/bin/env python3
"""
Execute only the Stanford Dogs detection + IoU eval cells (skips classifier training).
Run from project root: python run_localization_eval.py

Optional env:
  EVAL_MAX_IMAGES=500   — if set, overrides notebook max_images (subset after shuffle).
  If unset, uses whatever the notebook specifies (full dataset when max_images=None).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def main() -> None:
    nb_path = ROOT / "dog_detector.ipynb"
    nb = json.loads(nb_path.read_text())

    indices = [2, 45, 47, 50, 52, 56, 57, 58, 63]
    g: dict = {"__name__": "__main__"}
    for i in indices:
        cell = nb["cells"][i]
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Repo copy of resnet50_aug_progressive_fine_tune.pth may be full ImageNet (1000-way fc).
        if i == 45:
            src = src.replace(
                'model_path="resnet50_aug_progressive_fine_tune.pth"',
                'model_path="resnet50_noaug.pth"',
            )
            # Use Apple MPS or CUDA when available (notebook defaults to CPU if no CUDA).
            src = src.replace(
                'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
                'device = torch.device("cuda" if torch.cuda.is_available() else '
                '("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))',
            )
            print(
                "[runner] Using resnet50_noaug.pth (2-class). "
                "If aug weights are fixed, change default in cell 45.\n",
                flush=True,
            )
        if i == 47:
            n = os.environ.get("EVAL_MAX_IMAGES", "").strip()
            if n.isdigit():
                src = re.sub(
                    r"max_images=\s*None\s*,\s*shuffle_seed",
                    f"max_images={int(n)}, shuffle_seed",
                    src,
                )
                print(f"[runner] EVAL_MAX_IMAGES override = {n}\n", flush=True)
        print(f"\n--- Executing notebook cell {i} ---\n", flush=True)
        exec(compile(src, f"<notebook cell {i}>", "exec"), g, g)

    print("\n=== DONE ===\n")


if __name__ == "__main__":
    main()
