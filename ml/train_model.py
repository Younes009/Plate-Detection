"""
ML Training — Microservice ML (Python, indépendant de Spark).

Lit les features depuis la couche STAGING/ (READ 5 — écrite par ConsumerApp
à l'étape WRITE 4, AVANT toute prédiction), entraîne un RandomForestClassifier
sklearn sur les splits train/valid/test, exporte le modèle en ONNX (consommé
par le Consumer Scala) et en pickle (EXPORT 6 → dossier MODEL).

Usage :
    python ml/train_model.py --staging-dir data/staging --model-dir data/models

Dépendances : voir ml/requirements.txt
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

# Console Windows en cp1252 : évite UnicodeEncodeError sur les flèches
# quand stdout est redirigé (fichier, CI, tâche en arrière-plan).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Features géométriques — doivent correspondre exactement à ModelLoader.FEATURE_COLS (Scala)
FEATURE_COLS = ["bbox_area_norm", "aspect_ratio", "cx_norm", "cy_norm", "w_norm", "h_norm"]
READ_COLS = FEATURE_COLS + ["split", "class_id"]


def load_data(staging_dir: str) -> pd.DataFrame:
    """READ 5 — lit les Parquet de la couche staging (colonnes features uniquement,
    on ne charge pas les octets images en mémoire)."""
    files = glob.glob(os.path.join(staging_dir, "**", "*.parquet"), recursive=True)
    if not files:
        print(f"[Train] Aucun Parquet dans {staging_dir} — lancer Producer puis Consumer d'abord.")
        sys.exit(1)
    df = pd.concat([pd.read_parquet(f, columns=READ_COLS) for f in files], ignore_index=True)
    print(f"[Train] {len(df)} lignes lues depuis {staging_dir} ({len(files)} fichiers)")
    return df


def prepare_xy(df: pd.DataFrame):
    """
    Cible binaire : 1 si la ligne porte une bounding box YOLO (class_id non-null),
    0 sinon (image sans label = background).
    NB : le dataset est mono-classe, class_id vaut 0 pour toutes les plaques —
    la présence de la bbox EST le signal, pas la valeur de class_id.
    """
    df["target"] = df["class_id"].notna().astype(int)
    X = df[FEATURE_COLS].fillna(0.0).astype(np.float32)
    y = df["target"]
    return X, y


def split_by_column(df, X, y):
    """Utilise la colonne 'split' produite par le pipeline (train/valid/test/unknown)."""
    train_mask = df["split"].isin(["train", "unknown"])
    val_mask   = df["split"] == "valid"
    test_mask  = df["split"] == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    # Si les splits dédiés sont vides, fallback aléatoire 70/15/15
    if len(X_train) == 0 or len(X_test) == 0:
        from sklearn.model_selection import train_test_split
        strat = y if y.nunique() > 1 else None
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=strat)
        X_val,   X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)
        print("[Train] Splits par colonne vides → split aléatoire 70/15/15")

    print(f"[Train] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


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


def evaluate(clf, X_val, y_val, X_test, y_test):
    from sklearn.metrics import classification_report, accuracy_score
    names = ["background", "plate"]
    print("\n[Train] === Validation ===")
    print(classification_report(y_val, clf.predict(X_val), target_names=names, zero_division=0))
    print("\n[Train] === Test ===")
    print(classification_report(y_test, clf.predict(X_test), target_names=names, zero_division=0))
    print(f"[Train] Accuracy test : {accuracy_score(y_test, clf.predict(X_test)):.4f}")


def export_onnx(clf, model_dir: str) -> str:
    """
    EXPORT 6 — ONNX avec zipmap=False pour que output_probability soit un
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
    """EXPORT 6 — backup pickle."""
    pkl_path = os.path.join(model_dir, "plate_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[Train] Modèle pickle exporté → {pkl_path}")
    return pkl_path


def main():
    parser = argparse.ArgumentParser(description="Train plate detection model")
    parser.add_argument("--staging-dir", default="data/staging",
                        help="Couche staging écrite par le Consumer (WRITE 4)")
    parser.add_argument("--model-dir", default="data/models",
                        help="Dossier MODEL de sortie (chargé par le Consumer)")
    parser.add_argument("--format", default="both", choices=["onnx", "pickle", "both"],
                        help="Format d'export du modèle")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    df = load_data(args.staging_dir)

    X, y = prepare_xy(df)
    print(f"[Train] Distribution classes — 0 (background): {(y == 0).sum()}, 1 (plate): {(y == 1).sum()}")
    if y.nunique() < 2:
        print("[Train] ATTENTION : une seule classe présente — le modèle prédira "
              "toujours cette classe. Ajouter des images sans label pour des négatifs.")

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_column(df, X, y)

    print("\n[Train] Entraînement RandomForestClassifier...")
    clf = train(X_train, y_train)
    print(f"[Train] Feature importances : {dict(zip(FEATURE_COLS, clf.feature_importances_.round(3)))}")

    if len(X_val) > 0 and len(X_test) > 0:
        evaluate(clf, X_val, y_val, X_test, y_test)

    if args.format in ("onnx", "both"):
        export_onnx(clf, args.model_dir)
    if args.format in ("pickle", "both"):
        export_pickle(clf, args.model_dir)

    print(f"\n[Train] Modèle(s) disponible(s) dans : {args.model_dir}")
    print("[Train] Le Consumer charge plate_model.onnx automatiquement au prochain démarrage.")


if __name__ == "__main__":
    main()
