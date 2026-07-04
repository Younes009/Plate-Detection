"""
ML Training — Microservice ML.

Lit les features depuis OUTPUT/ (écrit par ConsumerApp),
entraîne un RandomForestClassifier sklearn sur les splits train/valid/test,
exporte le modèle en ONNX (pour le Consumer Scala) et en pickle (backup).

Usage :
    python train_model.py --output-dir /project/output --model-dir /project/models

Dépendances :
    pip install pyspark scikit-learn skl2onnx onnxruntime numpy pandas
"""

import argparse
import os
import pickle
import numpy as np

from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Features géométriques — doivent correspondre exactement à ModelLoader.FEATURE_COLS (Scala)
FEATURE_COLS = ["bbox_area_norm", "aspect_ratio", "cx_norm", "cy_norm", "w_norm", "h_norm"]


def load_data(spark: SparkSession, output_path: str):
    """Lit les features depuis outputPath/ (Parquet produit par ConsumerApp)."""
    df = spark.read.parquet(output_path)
    print(f"[Train] {df.count()} lignes lues depuis {output_path}")
    print(f"[Train] Colonnes : {df.columns}")
    return df.toPandas()


def prepare_xy(df):
    """
    Cible : 1 si la bbox est annotée comme une plaque (class_id non-null > 0), 0 sinon.
    Les images sans annotation (background) ont class_id = null.
    """
    df["target"] = ((df["class_id"].notna()) & (df["class_id"] > 0)).astype(int)

    # Remplissage des valeurs manquantes (images sans bbox)
    X = df[FEATURE_COLS].fillna(0.0).astype(np.float32)
    y = df["target"]
    return X, y


def split_by_column(df, X, y):
    """Utilise la colonne 'split' produite par le Producer (train/valid/test/unknown)."""
    train_mask = df["split"].isin(["train", "unknown"])
    val_mask   = df["split"] == "valid"
    test_mask  = df["split"] == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    # Si les splits dédiés sont vides, fallback aléatoire 70/15/15
    if len(X_train) == 0:
        from sklearn.model_selection import train_test_split
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        X_val,   X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)
        print("[Train] Splits par colonne vides → split aléatoire 70/15/15")

    print(f"[Train] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train(X_train, y_train):
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
    print("\n[Train] === Validation ===")
    print(classification_report(y_val, clf.predict(X_val), target_names=["background", "plate"]))
    print("\n[Train] === Test ===")
    print(classification_report(y_test, clf.predict(X_test), target_names=["background", "plate"]))
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[Train] Accuracy test : {acc:.4f}")


def export_onnx(clf, model_dir: str):
    """
    Exporte en ONNX avec zipmap=False pour que output_probability soit un Array[Array[Float]]
    (plus simple à consommer depuis Java/Scala avec ONNX Runtime).
    """
    n_features = len(FEATURE_COLS)
    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_types,
        options={type(clf): {"zipmap": False}},
    )
    onnx_path = os.path.join(model_dir, "plate_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[Train] Modèle ONNX exporté → {onnx_path}")

    # Vérification des noms de sorties (doit correspondre à ModelLoader.scala)
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path)
    print(f"[Train] Inputs  ONNX : {[i.name for i in sess.get_inputs()]}")
    print(f"[Train] Outputs ONNX : {[o.name for o in sess.get_outputs()]}")

    return onnx_path


def export_pickle(clf, model_dir: str):
    pkl_path = os.path.join(model_dir, "plate_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[Train] Modèle pickle exporté → {pkl_path}")
    return pkl_path


def main():
    parser = argparse.ArgumentParser(description="Train plate detection model")
    parser.add_argument("--output-dir", default="hdfs:///project/output/",
                        help="Chemin vers les données OUTPUT/ (Parquet du Consumer)")
    parser.add_argument("--model-dir",  default="./models",
                        help="Dossier local de sortie pour le modèle exporté")
    parser.add_argument("--format",     default="both", choices=["onnx", "pickle", "both"],
                        help="Format d'export du modèle")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("PlateDetection-ModelTraining")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    # Chargement des données
    df_pd = load_data(spark, args.output_dir)
    spark.stop()

    if df_pd.empty:
        print("[Train] Aucune donnée disponible, arrêt.")
        return

    # Préparation
    X, y = prepare_xy(df_pd)
    print(f"[Train] Distribution classes — 0 (background): {(y==0).sum()}, 1 (plate): {(y==1).sum()}")

    X_train, y_train, X_val, y_val, X_test, y_test = split_by_column(df_pd, X, y)

    # Entraînement
    print("\n[Train] Entraînement RandomForestClassifier...")
    clf = train(X_train, y_train)
    print(f"[Train] Feature importances : {dict(zip(FEATURE_COLS, clf.feature_importances_.round(3)))}")

    # Évaluation
    if len(X_val) > 0 and len(X_test) > 0:
        evaluate(clf, X_val, y_val, X_test, y_test)

    # Export
    if args.format in ("onnx", "both"):
        export_onnx(clf, args.model_dir)

    if args.format in ("pickle", "both"):
        export_pickle(clf, args.model_dir)

    print(f"\n[Train] Modèle(s) disponible(s) dans : {args.model_dir}")
    print("[Train] Copier plate_model.onnx dans hdfs:///project/models/ pour que le Consumer le charge.")


if __name__ == "__main__":
    main()
