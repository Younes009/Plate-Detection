"""
ML Training — Microservice ML (Python, indépendant de Spark). FLUX HEBDOMADAIRE.

Cycle de vie (une fois par semaine, après le Producer en mode batch) :
  1. READ 5   : lit la copie DEST/ml_dataset/ (arborescence train/test/val,
                images + labels YOLO) déposée par le Microservice 1.
  2. TRAIN    : RandomForestClassifier sur les features géométriques bbox.
  3. EVALUATE : accuracy / F1 / matrice de confusion sur les sets test et val.
  4. EXPORT 6 : plate_model.pkl (format contractuel) + plate_model.onnx
                (consommé par le Consumer Scala via ONNX Runtime) → MODEL/.
  5. HISTORIQUE : métriques versionnées dans MODEL/metrics/<version>.json,
                lues par l'onglet "Suivi de l'entraînement" du dashboard.

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

# Features géométriques — doivent correspondre exactement à ModelLoader.FEATURE_COLS (Scala)
FEATURE_COLS = ["bbox_area_norm", "aspect_ratio", "cx_norm", "cy_norm", "w_norm", "h_norm"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# Le dataset utilise "valid", le cahier des charges dit "val" : on accepte les deux.
SPLIT_ALIASES = {"train": "train", "test": "test", "val": "val", "valid": "val"}


def parse_label_file(path: str):
    """Parse un label YOLO : une ligne = class_id cx cy w h (normalisés)."""
    boxes = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                tokens = line.split()
                if len(tokens) >= 5:
                    cls, cx, cy, w, h = (float(t) for t in tokens[:5])
                    boxes.append((int(cls), cx, cy, w, h))
    except OSError:
        pass
    return boxes


def load_dataset(dataset_dir: str) -> pd.DataFrame:
    """
    READ 5 — lit la copie hebdomadaire DEST/ml_dataset/ (train/test/val).
    Une ligne par bounding box ; les images sans label produisent une ligne
    à features nulles (négatifs "background").
    """
    rows = []
    for entry in sorted(os.listdir(dataset_dir)):
        split = SPLIT_ALIASES.get(entry.lower())
        split_dir = os.path.join(dataset_dir, entry)
        if split is None or not os.path.isdir(split_dir):
            continue
        images = [p for p in glob.glob(os.path.join(split_dir, "**", "*"), recursive=True)
                  if p.lower().endswith(IMAGE_EXTS)]
        for img in images:
            name = os.path.splitext(os.path.basename(img))[0]
            label = (os.path.splitext(img)[0] + ".txt").replace(
                f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
            boxes = parse_label_file(label) if os.path.exists(label) else []
            if boxes:
                for cls, cx, cy, w, h in boxes:
                    rows.append({
                        "image_name": name, "split": split, "class_id": cls,
                        "cx_norm": cx, "cy_norm": cy, "w_norm": w, "h_norm": h,
                        "bbox_area_norm": w * h,
                        "aspect_ratio": (w / h) if h > 0 else np.nan,
                    })
            else:
                rows.append({"image_name": name, "split": split, "class_id": None,
                             "cx_norm": None, "cy_norm": None, "w_norm": None,
                             "h_norm": None, "bbox_area_norm": None, "aspect_ratio": None})

    if not rows:
        print(f"[Train] Aucune image dans {dataset_dir} — lancer d'abord "
              "le Producer en mode batch (python main.py --weekly).")
        sys.exit(1)

    df = pd.DataFrame(rows)
    counts = df.groupby("split")["image_name"].nunique().to_dict()
    print(f"[Train] {df['image_name'].nunique()} images / {len(df)} lignes lues "
          f"depuis {dataset_dir} — répartition : {counts}")
    return df


def prepare_xy(df: pd.DataFrame):
    """
    Cible binaire : 1 si la ligne porte une bounding box YOLO (class_id non-null),
    0 sinon (image sans label = background).
    NB : le dataset est mono-classe, class_id vaut 0 pour toutes les plaques —
    la présence de la bbox EST le signal, pas la valeur de class_id.
    """
    y = df["class_id"].notna().astype(int)
    X = df[FEATURE_COLS].fillna(0.0).astype(np.float32)
    return X, y


def split_sets(df, X, y):
    """Sépare selon l'arborescence train/test/val du dataset."""
    sets = {}
    for split in ("train", "val", "test"):
        mask = df["split"] == split
        sets[split] = (X[mask], y[mask])
    # Fallback aléatoire si l'arborescence est incomplète
    if len(sets["train"][0]) == 0 or len(sets["test"][0]) == 0:
        from sklearn.model_selection import train_test_split
        strat = y if y.nunique() > 1 else None
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
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


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


def export_onnx(clf, model_dir: str) -> str:
    """
    EXPORT 6 — ONNX avec zipmap=False pour que la sortie probabilités soit un
    Array[Array[Float]] (consommé tel quel par ModelLoader.scala via ONNX Runtime).
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
    # target_opset=21 : le Consumer utilise onnxruntime Java 1.20, qui refuse
    # les modèles opset 22+ (skl2onnx récent exporte en 22 par défaut)
    onnx_model = convert_sklearn(clf, initial_types=initial_types,
                                 target_opset={"": 21, "ai.onnx.ml": 3},
                                 options={type(clf): {"zipmap": False}})
    onnx_path = os.path.join(model_dir, "plate_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[Train] Modèle ONNX exporté → {onnx_path}")

    # Vérification des noms d'entrées/sorties (contrat avec ModelLoader.scala)
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path)
    print(f"[Train] Inputs  ONNX : {[i.name for i in sess.get_inputs()]}")
    print(f"[Train] Outputs ONNX : {[o.name for o in sess.get_outputs()]}")
    return onnx_path


def export_pickle(clf, model_dir: str) -> str:
    """EXPORT 6 — format .pkl contractuel du cahier des charges."""
    pkl_path = os.path.join(model_dir, "plate_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[Train] Modèle pickle exporté → {pkl_path}")
    return pkl_path


def write_metrics(model_dir: str, version: str, df: pd.DataFrame,
                  dataset_dir: str, results: dict, clf) -> str:
    """Historique versionné — consommé par l'onglet 2 du dashboard Streamlit."""
    metrics_dir = os.path.join(model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    payload = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": os.path.abspath(dataset_dir),
        "dataset_images": int(df["image_name"].nunique()),
        "dataset_rows": int(len(df)),
        "split_counts": {k: int(v) for k, v in
                         df.groupby("split")["image_name"].nunique().items()},
        "feature_importances": {c: round(float(v), 4) for c, v in
                                zip(FEATURE_COLS, clf.feature_importances_)},
        "metrics": results,
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
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    version = "v" + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[Train] Run hebdomadaire {version}")

    df = load_dataset(args.dataset_dir)

    X, y = prepare_xy(df)
    print(f"[Train] Distribution classes — 0 (background): {(y == 0).sum()}, 1 (plate): {(y == 1).sum()}")
    if y.nunique() < 2:
        print("[Train] ATTENTION : une seule classe présente — le modèle prédira "
              "toujours cette classe. Ajouter des images sans label pour des négatifs.")

    sets = split_sets(df, X, y)

    print("\n[Train] Entraînement RandomForestClassifier...")
    clf = train(*sets["train"])
    print(f"[Train] Feature importances : {dict(zip(FEATURE_COLS, clf.feature_importances_.round(3)))}")

    results = evaluate(clf, sets)

    if args.format in ("pickle", "both"):
        export_pickle(clf, args.model_dir)
    if args.format in ("onnx", "both"):
        export_onnx(clf, args.model_dir)
    write_metrics(args.model_dir, version, df, args.dataset_dir, results, clf)

    print(f"\n[Train] Modèle {version} disponible dans : {args.model_dir}")
    print("[Train] Le Consumer charge plate_model.onnx automatiquement au prochain démarrage.")


if __name__ == "__main__":
    main()
