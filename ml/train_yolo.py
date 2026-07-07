"""
ML Training — YOLO plate detector (Python, Spark-independent). Weekly job.

Fine-tunes YOLOv8n on the weekly copy DEST/ml_dataset (native YOLO layout
{split}/{images,labels}, labels used HERE only — never in the stream),
evaluates IoU against ground truth, exports plate_locator.onnx for the Scala
Consumer and versions metrics for the dashboard.

ONNX contract (ModelLoader.scala):
    input  "images"  [1, 3, S, S]  RGB /255 CHW
    output "output0" [1, 4+k, N]   rows 0-3 = cx, cy, w, h in PIXELS of S,
                                   rows 4.. = per-class scores

Usage:
    python ml/train_yolo.py                    # 3 epochs, 256px
    python ml/train_yolo.py --epochs 10
    python ml/train_yolo.py --skip-train       # re-export + eval only
"""

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_SIZE = 256
RUNS_DIR = os.path.join(os.path.dirname(__file__), "yolo_runs")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ─── Shared helpers (also used by ml/tests) ──────────────────────────────────

def largest_bbox(image_path: str):
    """Largest-area YOLO bbox (cx, cy, w, h normalized) or None if unlabeled."""
    label = (os.path.splitext(image_path)[0] + ".txt").replace(
        f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    best, best_area = None, -1.0
    try:
        with open(label, encoding="utf-8") as f:
            for line in f:
                t = line.split()
                if len(t) >= 5:
                    cx, cy, w, h = (float(v) for v in t[1:5])
                    if w * h > best_area:
                        best, best_area = (cx, cy, w, h), w * h
    except OSError:
        return None
    return best


def iou_yolo(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Vectorized IoU between YOLO boxes (cx, cy, w, h)."""
    def corners(b):
        return (b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2)
    px0, py0, px1, py1 = corners(pred)
    tx0, ty0, tx1, ty1 = corners(truth)
    ix = np.clip(np.minimum(px1, tx1) - np.maximum(px0, tx0), 0, None)
    iy = np.clip(np.minimum(py1, ty1) - np.maximum(py0, ty0), 0, None)
    inter = ix * iy
    union = pred[:, 2] * pred[:, 3] + truth[:, 2] * truth[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


# ─── Training / evaluation / export ──────────────────────────────────────────

def write_dataset_yaml(dataset_dir: str) -> str:
    root = os.path.abspath(dataset_dir)
    val = "valid" if os.path.isdir(os.path.join(root, "valid")) else "val"
    yaml_path = os.path.join(RUNS_DIR, "plate_dataset.yaml")
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {root}\n"
                f"train: train/images\n"
                f"val: {val}/images\n"
                f"test: test/images\n"
                f"names:\n  0: license-plate\n")
    return yaml_path


def dataset_counts(dataset_dir: str) -> dict:
    counts = {}
    for split in ("train", "valid", "val", "test"):
        img_dir = os.path.join(dataset_dir, split, "images")
        if os.path.isdir(img_dir):
            key = "val" if split in ("val", "valid") else split
            counts[key] = len([p for p in os.listdir(img_dir)
                               if p.lower().endswith(IMAGE_EXTS)])
    if not counts:
        print(f"[YOLO] no dataset in {dataset_dir} — run the batch Producer "
              "first (python main.py --weekly).")
        sys.exit(1)
    return counts


def train(dataset_dir: str, epochs: int, imgsz: int):
    from ultralytics import YOLO

    yaml_path = write_dataset_yaml(dataset_dir)
    model = YOLO("yolov8n.pt")  # COCO-pretrained, single-class fine-tune
    print(f"[YOLO] fine-tuning yolov8n — {epochs} epochs, {imgsz}px, CPU")
    model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=16,
                device="cpu", workers=0, project=RUNS_DIR, name="plate",
                exist_ok=True, verbose=True, plots=False)
    return model


def best_weights() -> str:
    path = os.path.join(RUNS_DIR, "plate", "weights", "best.pt")
    if not os.path.exists(path):
        print(f"[YOLO] weights not found: {path} — run without --skip-train.")
        sys.exit(1)
    return path


def evaluate(dataset_dir: str, n_samples: int) -> dict:
    """IoU of the best predicted box vs the largest ground-truth bbox."""
    from ultralytics import YOLO

    model = YOLO(best_weights())
    results = {}
    for split in ("valid", "val", "test"):
        img_dir = os.path.join(dataset_dir, split, "images")
        if not os.path.isdir(img_dir):
            continue
        images = [p for p in glob.glob(os.path.join(img_dir, "*"))
                  if p.lower().endswith(IMAGE_EXTS)][:n_samples]
        preds, truths = [], []
        for path in images:
            truth = largest_bbox(path)
            if truth is None:
                continue
            r = model.predict(path, imgsz=IMG_SIZE, device="cpu",
                              verbose=False, conf=0.01)[0]
            if len(r.boxes) == 0:
                preds.append((0.5, 0.5, 0.0, 0.0))  # no detection
            else:
                b = r.boxes[int(r.boxes.conf.argmax())]
                preds.append(tuple(b.xywhn[0].tolist()))
            truths.append(truth)
        if not truths:
            continue
        ious = iou_yolo(np.array(preds, np.float32), np.array(truths, np.float32))
        key = "val" if split in ("val", "valid") else split
        results[key] = {
            "n_samples": len(truths),
            "iou_mean": round(float(ious.mean()), 4),
            "iou_median": round(float(np.median(ious)), 4),
            "pct_iou_over_50": round(float((ious >= 0.5).mean()), 4),
            "model": "yolov8n",
        }
        print(f"[YOLO] {key}: IoU mean={results[key]['iou_mean']} "
              f"median={results[key]['iou_median']} "
              f"IoU>=0.5={results[key]['pct_iou_over_50']:.0%} (n={len(truths)})")
    return results


def export_onnx(model_dir: str) -> str:
    """Static 256px export, opset 17 (onnxruntime Java 1.20)."""
    from ultralytics import YOLO

    model = YOLO(best_weights())
    exported = model.export(format="onnx", imgsz=IMG_SIZE, opset=17,
                            dynamic=False, device="cpu")
    target = os.path.join(model_dir, "plate_locator.onnx")
    shutil.copy2(exported, target)

    import onnxruntime as rt
    sess = rt.InferenceSession(target)
    print(f"[YOLO] plate_locator.onnx exported — inputs "
          f"{[(i.name, i.shape) for i in sess.get_inputs()]} / outputs "
          f"{[(o.name, o.shape) for o in sess.get_outputs()]}")
    return target


def write_metrics(model_dir: str, version: str, dataset_dir: str,
                  counts: dict, locator_results: dict) -> str:
    """Versioned run history consumed by dashboard tab 2."""
    metrics_dir = os.path.join(model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    payload = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": os.path.abspath(dataset_dir),
        "dataset_images": int(sum(counts.values())),
        "split_counts": counts,
        "model": "yolov8n",
        "img_size": IMG_SIZE,
        "locator": locator_results,
    }
    path = os.path.join(metrics_dir, f"{version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[YOLO] metrics versioned -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="YOLOv8n plate detector — weekly training")
    ap.add_argument("--dataset-dir", default="data/dest/ml_dataset")
    ap.add_argument("--model-dir", default="data/models")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--samples", type=int, default=300, help="eval images per split")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    version = "v" + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[YOLO] weekly run {version}")
    counts = dataset_counts(args.dataset_dir)
    print(f"[YOLO] dataset splits: {counts}")

    if not args.skip_train:
        train(args.dataset_dir, args.epochs, args.imgsz)
    results = evaluate(args.dataset_dir, args.samples)
    export_onnx(args.model_dir)
    write_metrics(args.model_dir, version, args.dataset_dir, counts, results)
    print("[YOLO] done — the Consumer loads plate_locator.onnx on next start.")


if __name__ == "__main__":
    main()
