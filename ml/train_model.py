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

# Côté image : S×S RGB aplati CHW → 3*S*S features.
# ModelLoader.scala DÉDUIT S de la forme d'entrée du modèle ONNX : changer
# IMG_SIDE ici suffit, aucun changement côté Scala.
IMG_SIDE = 32
N_FEATURES = 3 * IMG_SIDE * IMG_SIDE

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# Le dataset utilise "valid", le cahier des charges dit "val" : on accepte les deux.
SPLIT_ALIASES = {"train": "train", "test": "test", "val": "val", "valid": "val"}


def has_plate_label(image_path: str) -> bool:
    """Vrai si un label YOLO non vide existe pour cette image (≥ 1 bbox valide)."""
    label = (os.path.splitext(image_path)[0] + ".txt").replace(
        f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    try:
        with open(label, encoding="utf-8") as f:
            return any(len(line.split()) >= 5 for line in f)
    except OSError:
        return False


def image_to_features(image_path: str) -> np.ndarray:
    """Image → vecteur float32 [3*S*S] : resize S×S RGB, /255, aplati CHW.
    Doit rester identique au prétraitement de ModelLoader.scala (toFloatCHW)."""
    from PIL import Image
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("RGB").resize((IMG_SIDE, IMG_SIDE)),
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
                rows.append({"image_path": img,
                             "image_name": os.path.splitext(os.path.basename(img))[0],
                             "split": split,
                             "target": int(has_plate_label(img))})
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
    EXPORT — ONNX avec zipmap=False pour que la sortie probabilités soit un
    Array[Array[Float]] (consommé tel quel par ModelLoader.scala via ONNX Runtime).
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("float_input", FloatTensorType([None, N_FEATURES]))]
    # target_opset=21 : le Consumer utilise onnxruntime Java 1.20, qui refuse
    # les modèles opset 22+ (skl2onnx récent exporte en 22 par défaut)
    onnx_model = convert_sklearn(clf, initial_types=initial_types,
                                 target_opset={"": 21, "ai.onnx.ml": 3},
                                 options={type(clf): {"zipmap": False}})
    onnx_path = os.path.join(model_dir, "plate_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[Train] Modèle ONNX exporté → {onnx_path}")

    # Vérification du contrat avec ModelLoader.scala (noms + forme d'entrée)
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path)
    print(f"[Train] Inputs  ONNX : {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"[Train] Outputs ONNX : {[o.name for o in sess.get_outputs()]}")
    return onnx_path


def export_pickle(clf, model_dir: str) -> str:
    """EXPORT — backup pickle."""
    pkl_path = os.path.join(model_dir, "plate_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[Train] Modèle pickle exporté → {pkl_path}")
    return pkl_path


def write_metrics(model_dir: str, version: str, df: pd.DataFrame,
                  dataset_dir: str, results: dict) -> str:
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
    y = df["target"].to_numpy()
    print(f"[Train] Distribution classes — 0 (background): {(y == 0).sum()}, 1 (plate): {(y == 1).sum()}")
    if len(np.unique(y)) < 2:
        print("[Train] ATTENTION : une seule classe présente — le modèle prédira "
              "toujours cette classe. Ajouter des images sans label pour des négatifs.")

    X = build_features(df)
    sets = split_sets(df, X, y)

    print("\n[Train] Entraînement RandomForestClassifier...")
    clf = train(*sets["train"])

    results = evaluate(clf, sets)

    if args.format in ("pickle", "both"):
        export_pickle(clf, args.model_dir)
    if args.format in ("onnx", "both"):
        export_onnx(clf, args.model_dir)
    write_metrics(args.model_dir, version, df, args.dataset_dir, results)

    print(f"\n[Train] Modèle {version} disponible dans : {args.model_dir}")
    print("[Train] Le Consumer charge plate_model.onnx automatiquement au prochain démarrage.")


if __name__ == "__main__":
    main()
