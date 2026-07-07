"""
ML Training — Microservice ML (Python, indépendant de Spark). FLUX HEBDOMADAIRE.

Script ÉPHÉMÈRE déclenché une fois par semaine, juste après le Producer en
mode batch :
  1. READ      : lit la copie DEST/ml_dataset/ (arborescence train/test/val,
                 images + labels YOLO — les labels ne servent QU'ICI, jamais
                 dans le stream d'inférence).
  2. FEATURES  : chaque image est réduite en S×S RGB, aplatie CHW, /255 —
                 EXACTEMENT le prétraitement appliqué par ModelLoader.scala
                 au moment du scoring (le stream n'a pas de labels, le modèle
                 doit prédire depuis les pixels bruts).
  3. TRAIN     : RandomForestClassifier ; cible = présence d'un label YOLO
                 non vide (plaque) vs absence (background).
  4. EVALUATE  : accuracy / F1 / matrice de confusion sur test et val.
  5. EXPORT    : plate_model.onnx (consommé par le Consumer Scala) +
                 plate_model.pkl (backup) → MODEL/ ; métriques versionnées
                 dans MODEL/metrics/<version>.json (onglet 2 du dashboard).

Usage :
    python ml/train_model.py --dataset-dir data/dest/ml_dataset --model-dir data/models

Planification hebdomadaire (exemples) :
    Windows : schtasks /Create /SC WEEKLY /TN PlateRetrain /TR "python main.py --weekly"
    Linux   : 0 3 * * 1  cd /opt/pipeline && python main.py --weekly
"""

import argparse
import glob
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Console Windows en cp1252 : évite UnicodeEncodeError sur les flèches
# quand stdout est redirigé (fichier, CI, tâche planifiée).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Classifieur sklearn : S×S RGB aplati CHW → 3*S*S features.
# ModelLoader.scala DÉDUIT S de la forme d'entrée du modèle ONNX : changer
# IMG_SIDE ici suffit, aucun changement côté Scala.
IMG_SIDE = 32
N_FEATURES = 3 * IMG_SIDE * IMG_SIDE

# Localisateur CNN (PyTorch) : entrée [batch, 3, LOC_SIDE, LOC_SIDE], /255,
# resize BILINEAR — même prétraitement que toFloatCHW côté Scala (le rank 4
# de la forme d'entrée ONNX déclenche le chemin CNN dans ModelLoader).
LOC_SIDE = 128

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# Le dataset utilise "valid", le cahier des charges dit "val" : on accepte les deux.
SPLIT_ALIASES = {"train": "train", "test": "test", "val": "val", "valid": "val"}


def largest_bbox(image_path: str):
    """
    Bbox YOLO de plus grande aire pour cette image (cx, cy, w, h normalisés),
    None si aucun label. Cible du LOCALISATEUR : une plaque par image (la plus
    visible) — suffisant pour tracer la boîte prédite au dashboard.
    """
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


def image_to_features(image_path: str) -> np.ndarray:
    """Image → vecteur float32 [3*S*S] : resize S×S RGB, /255, aplati CHW.
    Doit rester identique au prétraitement de ModelLoader.scala (toFloatCHW).
    BILINEAR explicite : le défaut PIL est BICUBIC, alors que le resize Scala
    (AWT) est bilinéaire — sans ça, distribution train ≠ inférence."""
    from PIL import Image
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("RGB").resize((IMG_SIDE, IMG_SIDE),
                                                   Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1).ravel()  # HWC → CHW, aplati


def load_dataset(dataset_dir: str) -> pd.DataFrame:
    """
    READ — parcourt la copie hebdo DEST/ml_dataset/ (une ligne par image).

    Arborescence IMPÉRATIVE (chaque split est étanche : ses labels ne
    concernent que ses propres images) :
        ml_dataset/{train,test,val}/images/*.jpg
        ml_dataset/{train,test,val}/labels/*.txt
    """
    rows = []
    for entry in sorted(os.listdir(dataset_dir)):
        split = SPLIT_ALIASES.get(entry.lower())
        split_dir = os.path.join(dataset_dir, entry)
        if split is None or not os.path.isdir(split_dir):
            continue
        images_dir = os.path.join(split_dir, "images")
        if not os.path.isdir(images_dir):
            print(f"[Train] ERREUR : sous-répertoire images/ manquant dans {split_dir} "
                  "(arborescence attendue : <split>/images + <split>/labels).")
            sys.exit(1)
        if not os.path.isdir(os.path.join(split_dir, "labels")):
            print(f"[Train] ATTENTION : pas de labels/ dans {split_dir} — "
                  "toutes les images de ce split seront des négatifs (background).")
        for img in glob.glob(os.path.join(images_dir, "**", "*"), recursive=True):
            if img.lower().endswith(IMAGE_EXTS):
                bbox = largest_bbox(img)
                rows.append({"image_path": img,
                             "image_name": os.path.splitext(os.path.basename(img))[0],
                             "split": split,
                             "target": int(bbox is not None),
                             "cx": bbox[0] if bbox else np.nan,
                             "cy": bbox[1] if bbox else np.nan,
                             "w":  bbox[2] if bbox else np.nan,
                             "h":  bbox[3] if bbox else np.nan})
    if not rows:
        print(f"[Train] Aucune image dans {dataset_dir} — lancer d'abord "
              "le Producer en mode batch (python main.py --weekly).")
        sys.exit(1)

    df = pd.DataFrame(rows)
    counts = df.groupby("split")["image_name"].nunique().to_dict()
    print(f"[Train] {len(df)} images lues depuis {dataset_dir} — répartition : {counts}")
    return df


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Vectorise toutes les images (S×S → mémoire maîtrisée : 3*S*S float32/image)."""
    print(f"[Train] Extraction des features pixels ({IMG_SIDE}×{IMG_SIDE} RGB, "
          f"{N_FEATURES} dims) ...")
    X = np.zeros((len(df), N_FEATURES), dtype=np.float32)
    for i, path in enumerate(df["image_path"].to_numpy()):
        try:
            X[i] = image_to_features(path)
        except Exception:
            pass  # image illisible → vecteur nul, sera un point de bruit
        if (i + 1) % 2000 == 0:
            print(f"[Train]   ... {i + 1}/{len(df)} images vectorisées")
    return X


def split_sets(df, X, y):
    """Sépare selon l'arborescence train/test/val du dataset."""
    sets = {}
    for split in ("train", "val", "test"):
        mask = (df["split"] == split).to_numpy()
        sets[split] = (X[mask], y[mask])
    # Fallback aléatoire si l'arborescence est incomplète
    if len(sets["train"][0]) == 0 or len(sets["test"][0]) == 0:
        from sklearn.model_selection import train_test_split
        strat = y if len(np.unique(y)) > 1 else None
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                    random_state=42, stratify=strat)
        X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                    random_state=42)
        sets = {"train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_te, y_te)}
        print("[Train] Arborescence incomplète → split aléatoire 70/15/15")
    print("[Train] " + ", ".join(f"{k}={len(v[0])}" for k, v in sets.items()))
    return sets


def train(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def _locator_dataset_cls():
    """Dataset torch : décode/resize à la volée (pas de matrice géante en RAM)."""
    import torch
    from torch.utils.data import Dataset

    class PlateBoxDataset(Dataset):
        def __init__(self, paths, boxes):
            self.paths, self.boxes = list(paths), np.asarray(boxes, dtype=np.float32)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            from PIL import Image
            with Image.open(self.paths[i]) as img:
                arr = np.asarray(
                    img.convert("RGB").resize((LOC_SIDE, LOC_SIDE),
                                              Image.Resampling.BILINEAR),
                    dtype=np.float32) / 255.0
            return torch.from_numpy(arr.transpose(2, 0, 1)), torch.from_numpy(self.boxes[i])

    return PlateBoxDataset


def _locator_net():
    """Petit CNN de régression bbox : conv stride-2 empilées + GAP + tête FC.
    Sortie sigmoïde 4 = (cx, cy, w, h) normalisés [0,1]."""
    import torch
    from torch import nn

    class LocatorNet(nn.Module):
        def __init__(self):
            super().__init__()
            def block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True))
            self.features = nn.Sequential(
                block(3, 16), block(16, 32), block(32, 64), block(64, 128),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 4))

        def forward(self, x):
            return torch.sigmoid(self.head(self.features(x)))

    return LocatorNet()


def train_cnn_locator(df: pd.DataFrame, epochs: int):
    """
    LOCALISATEUR CNN : images AVEC plaque uniquement, cible = bbox YOLO de plus
    grande aire. SmoothL1 sur les 4 coordonnées, suivi IoU sur le split val.
    """
    import torch
    from torch.utils.data import DataLoader

    labeled = df[df["target"] == 1]
    tr = labeled[labeled["split"] == "train"]
    va = labeled[labeled["split"] == "val"]
    if len(tr) == 0:
        print("[Train] Aucune bbox dans le split train — localisateur non entraîné.")
        return None

    ds_cls = _locator_dataset_cls()
    boxes_of = lambda d: d[["cx", "cy", "w", "h"]].to_numpy(dtype=np.float32)
    train_dl = DataLoader(ds_cls(tr["image_path"], boxes_of(tr)),
                          batch_size=64, shuffle=True)
    val_dl = (DataLoader(ds_cls(va["image_path"], boxes_of(va)), batch_size=128)
              if len(va) else None)

    torch.manual_seed(42)
    model = _locator_net()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()

    print(f"[Train] CNN localisateur : {len(tr)} images train, {epochs} epochs, "
          f"{LOC_SIDE}×{LOC_SIDE}")
    for epoch in range(1, epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in train_dl:
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            total += float(loss) * len(x)
            seen += len(x)
        msg = f"[Train]   epoch {epoch}/{epochs} — loss={total / seen:.4f}"
        if val_dl is not None:
            model.eval()
            ious = []
            with torch.no_grad():
                for x, y in val_dl:
                    ious.append(iou_yolo(model(x).numpy(), y.numpy()))
            msg += f" — IoU val={np.concatenate(ious).mean():.4f}"
        print(msg)
    model.eval()
    return model


def evaluate_locator(model, df: pd.DataFrame) -> dict:
    """IoU / MAE du CNN sur les splits val et test (images avec plaque)."""
    import torch
    from torch.utils.data import DataLoader

    ds_cls = _locator_dataset_cls()
    results = {}
    for split in ("val", "test"):
        part = df[(df["split"] == split) & (df["target"] == 1)]
        if len(part) == 0:
            continue
        dl = DataLoader(ds_cls(part["image_path"],
                               part[["cx", "cy", "w", "h"]].to_numpy(dtype=np.float32)),
                        batch_size=128)
        preds, truths = [], []
        with torch.no_grad():
            for x, y in dl:
                preds.append(model(x).numpy())
                truths.append(y.numpy())
        pred, truth = np.concatenate(preds), np.concatenate(truths)
        ious = iou_yolo(pred, truth)
        results[split] = {
            "n_samples": int(len(part)),
            "iou_mean": round(float(ious.mean()), 4),
            "iou_median": round(float(np.median(ious)), 4),
            "pct_iou_over_50": round(float((ious >= 0.5).mean()), 4),
            "mae": round(float(np.abs(pred - truth).mean()), 4),
        }
        print(f"[Train] locator {split} : IoU moy={results[split]['iou_mean']} "
              f"méd={results[split]['iou_median']} "
              f"IoU≥0.5={results[split]['pct_iou_over_50']:.0%} (n={len(part)})")
    return results


def iou_yolo(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """IoU vectorisé entre bbox YOLO (cx, cy, w, h) prédites et vérité terrain."""
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


def evaluate(clf, sets) -> dict:
    """Accuracy / F1 / matrice de confusion sur test et val — persistées ensuite."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    results = {}
    for split in ("val", "test"):
        X, y = sets[split]
        if len(X) == 0:
            continue
        pred = clf.predict(X)
        results[split] = {
            "n_samples": int(len(y)),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1_score": round(float(f1_score(y, pred, zero_division=0)), 4),
            # labels fixés : matrice 2x2 même si une classe est absente du set
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        }
        print(f"[Train] {split} : accuracy={results[split]['accuracy']} "
              f"f1={results[split]['f1_score']} (n={results[split]['n_samples']})")
    return results


def _export_onnx(model, model_dir: str, filename: str, options=None) -> str:
    """
    EXPORT ONNX générique. target_opset=21 : le Consumer utilise onnxruntime
    Java 1.20, qui refuse les modèles opset 22+ (skl2onnx récent par défaut).
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types,
                                 target_opset={"": 21, "ai.onnx.ml": 3},
                                 options=options)
    onnx_path = os.path.join(model_dir, filename)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Vérification du contrat avec ModelLoader.scala (noms + formes)
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path)
    print(f"[Train] {filename} exporté — inputs "
          f"{[(i.name, i.shape) for i in sess.get_inputs()]} / outputs "
          f"{[(o.name, o.shape) for o in sess.get_outputs()]}")
    return onnx_path


def export_classifier_onnx(clf, model_dir: str) -> str:
    # zipmap=False : sortie probabilités en Array[Array[Float]] côté Java
    return _export_onnx(clf, model_dir, "plate_model.onnx",
                        options={type(clf): {"zipmap": False}})


def export_locator_onnx(model, model_dir: str) -> str:
    """
    EXPORT CNN → ONNX. Entrée rank 4 [batch, 3, S, S] : c'est cette forme qui
    fait basculer ModelLoader.scala sur le prétraitement CNN. Opset 17
    (compatible onnxruntime Java 1.20).
    """
    import torch

    onnx_path = os.path.join(model_dir, "plate_locator.onnx")
    dummy = torch.zeros(1, 3, LOC_SIDE, LOC_SIDE)
    torch.onnx.export(model, (dummy,), onnx_path,
                      input_names=["image_input"], output_names=["bbox"],
                      dynamic_axes={"image_input": {0: "batch"}, "bbox": {0: "batch"}},
                      opset_version=17, dynamo=False)

    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path)
    print(f"[Train] plate_locator.onnx exporté — inputs "
          f"{[(i.name, i.shape) for i in sess.get_inputs()]} / outputs "
          f"{[(o.name, o.shape) for o in sess.get_outputs()]}")
    return onnx_path


def export_pickle(model, model_dir: str, filename: str) -> str:
    """EXPORT — backup pickle."""
    pkl_path = os.path.join(model_dir, filename)
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[Train] {filename} exporté")
    return pkl_path


def write_metrics(model_dir: str, version: str, df: pd.DataFrame,
                  dataset_dir: str, results: dict, locator_results: dict) -> str:
    """Historique versionné — consommé par l'onglet 2 du dashboard Streamlit."""
    metrics_dir = os.path.join(model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    payload = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": os.path.abspath(dataset_dir),
        "dataset_images": int(len(df)),
        "split_counts": {k: int(v) for k, v in
                         df.groupby("split")["image_name"].nunique().items()},
        "img_side": IMG_SIDE,
        "metrics": results,
        "locator": locator_results,
    }
    path = os.path.join(metrics_dir, f"{version}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[Train] Métriques versionnées → {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Réentraînement hebdomadaire du modèle")
    parser.add_argument("--dataset-dir", default="data/dest/ml_dataset",
                        help="Copie hebdo du dataset (déposée par le Producer batch)")
    parser.add_argument("--model-dir", default="data/models",
                        help="Dossier MODEL de sortie (chargé par le Consumer)")
    parser.add_argument("--format", default="both", choices=["onnx", "pickle", "both"],
                        help="Format d'export du modèle")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs d'entraînement du localisateur CNN")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    version = "v" + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[Train] Run hebdomadaire {version}")

    df = load_dataset(args.dataset_dir)
    y = df["target"].to_numpy()
    print(f"[Train] Distribution classes — 0 (background): {(y == 0).sum()}, 1 (plate): {(y == 1).sum()}")
    if len(np.unique(y)) < 2:
        print("[Train] ATTENTION : une seule classe présente — le modèle prédira "
              "toujours cette classe. Ajouter des images sans label pour des négatifs.")

    X = build_features(df)
    sets = split_sets(df, X, y)

    print("\n[Train] Entraînement RandomForestClassifier (plaque / background)...")
    clf = train(*sets["train"])
    results = evaluate(clf, sets)

    # ─── Localisateur CNN : image → bbox (cx, cy, w, h) autour de la plaque ───
    print("\n[Train] Entraînement du localisateur CNN (régression bbox)...")
    locator = train_cnn_locator(df, epochs=args.epochs)
    locator_results = evaluate_locator(locator, df) if locator is not None else {}

    if args.format in ("pickle", "both"):
        export_pickle(clf, args.model_dir, "plate_model.pkl")
        if locator is not None:
            import torch
            torch.save(locator.state_dict(),
                       os.path.join(args.model_dir, "plate_locator.pt"))
            print("[Train] plate_locator.pt exporté (state_dict PyTorch)")
    if args.format in ("onnx", "both"):
        export_classifier_onnx(clf, args.model_dir)
        if locator is not None:
            export_locator_onnx(locator, args.model_dir)
    write_metrics(args.model_dir, version, df, args.dataset_dir, results, locator_results)

    print(f"\n[Train] Modèle {version} disponible dans : {args.model_dir}")
    print("[Train] Le Consumer charge plate_model.onnx automatiquement au prochain démarrage.")


if __name__ == "__main__":
    main()
