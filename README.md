# ResNet50 Sliding-Window Dog Detector

A two-phase computer vision pipeline: train and compare three dog-vs-cat classifiers, then reuse the strongest one — unmodified — as a sliding-window object detector that localizes dogs in full scenes and is scored with IoU against ground-truth boxes.

## Pipeline

**Phase 1 — Classification.** Three approaches are trained and compared on the same train/test split:

| Method | Accuracy | Training time |
|---|---|---|
| HOG + SVM | 64.6% | ~10 min (CPU) |
| PCA + SVM | 68.9% | ~2 min (CPU) |
| ResNet50 (progressive unfreeze + augmentation) | 99.7% | ~12 min (GPU) |

**Phase 2 — Detection & localization.** The fine-tuned ResNet50 classifier is wrapped in a `score_patches()` interface and reused, as-is, as a sliding-window detector:

1. **Image pyramid** — scale factor 1.25, down to a 128px minimum side.
2. **Sliding windows** — square windows of `{96, 128, 160, 192}` px per pyramid level, stride 32, each resized to 224×224 before scoring.
3. **Threshold + NMS** — keep patches scoring ≥ 0.7, then greedy non-maximum suppression at IoU 0.4.

Evaluated on 200 images from the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/):

| Metric | Value |
|---|---|
| Mean IoU | 0.357 |
| Median IoU | 0.319 |
| IoU ≥ 0.5 | 25.5% |
| IoU ≥ 0.3 | 52.5% |

Sample outputs live in `outputs/`: `outputs/detector_demo/` (candidate → threshold → NMS visualizations), `outputs/qualitative/` (success/borderline/failure cases), `outputs/best_worst/` (top/bottom-5 IoU predictions), and `outputs/per_image_iou_results.csv` (full per-image results).

## Project structure

```
dog_detector.ipynb               # full pipeline: classification, detection, evaluation
run_localization_eval.py         # headless runner for just the detection + IoU eval cells
export_best_worst_qualitative.py # renders GT/prediction overlays for the best/worst IoU cases
resnet50_noaug.pth                # trained classifier weights used by the detector pipeline
outputs/                          # qualitative and quantitative evaluation artifacts
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
```

- Open `dog_detector.ipynb` for the full walkthrough (classifier training + comparison, then detection + evaluation).
- `python run_localization_eval.py` runs just the detection/IoU-evaluation cells against the included `resnet50_noaug.pth` checkpoint, skipping classifier training.
- `python export_best_worst_qualitative.py` regenerates the best/worst overlay images in `outputs/best_worst/` from `outputs/per_image_iou_results.csv`.

Expects a `data/` directory (not included, due to size) with a Kaggle-style `data/train/train/`, `data/test/test/` dogs-vs-cats split and `data/stanford_dogs/` (images + PASCAL-VOC XML annotations) for the detection phase.

## Contributors

Tarek Namani, Hugues Salmon, Sanad Abu Baker, Paul Litscher, Boris Vassilev
