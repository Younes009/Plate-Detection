"""
Dashboard Streamlit — LECTURE SEULE.
Lit les Parquet locaux produits par ConsumerApp via pandas+pyarrow.
"""

import os
import glob
import pandas as pd
import streamlit as st

_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH      = os.path.join(_BASE, "output")
PREDICTIONS_PATH = os.path.join(_BASE, "output", "predictions")

DISPLAY_COLUMNS = [
    "image_name", "split", "label", "class_id", "num_boxes",
    "bbox_area_norm", "aspect_ratio", "bbox_size_category",
    "center_x_quadrant", "center_y_quadrant",
    "quality_flag", "prediction", "confidence_score", "processed_at",
]

PREDICTIONS_COLUMNS = ["image_path", "prediction", "confidence_score", "processed_at"]


def _parquet_files(directory: str):
    return glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True)


@st.cache_data(ttl=60)
def load_output() -> pd.DataFrame:
    files = _parquet_files(OUTPUT_PATH)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    return df[cols]


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    files = _parquet_files(PREDICTIONS_PATH)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cols = [c for c in PREDICTIONS_COLUMNS if c in df.columns]
    return df[cols]


# ─── Layout ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Plate Detection Dashboard", layout="wide")
st.title("Dashboard — Plate Detection Pipeline")
st.caption(f"Source : {OUTPUT_PATH}  |  Cache TTL 60 s")

if st.button("Rafraîchir maintenant"):
    st.cache_data.clear()

try:
    data = load_output()
except Exception as exc:
    st.error(f"Impossible de lire {OUTPUT_PATH} : {exc}")
    st.stop()

if data.empty:
    st.info("Aucune donnée dans output/ pour le moment. Lancer le Producer puis le Consumer.")
    st.stop()

# ─── Filtres sidebar ─────────────────────────────────────────────────────────

st.sidebar.header("Filtres")
splits        = sorted(data["split"].dropna().unique().tolist()) if "split" in data.columns else []
labels        = sorted(data["label"].dropna().unique().tolist()) if "label" in data.columns else []
quality_flags = sorted(data["quality_flag"].dropna().unique().tolist()) if "quality_flag" in data.columns else []
predictions   = sorted(data["prediction"].dropna().unique().tolist()) if "prediction" in data.columns else []

selected_splits  = st.sidebar.multiselect("Split",          splits,        default=splits)
selected_labels  = st.sidebar.multiselect("Classe (label)", labels,        default=labels)
selected_quality = st.sidebar.multiselect("Quality flag",   quality_flags, default=quality_flags)
selected_preds   = st.sidebar.multiselect("Prédiction",     predictions,   default=predictions)
min_conf = st.sidebar.slider("Confidence score minimum", 0.0, 1.0, 0.0, 0.05)

mask = (
    data["split"].isin(selected_splits)
    & data["quality_flag"].isin(selected_quality)
    & data["prediction"].isin(selected_preds)
    & (data["confidence_score"] >= min_conf)
)
if selected_labels and "label" in data.columns:
    mask &= data["label"].isin(selected_labels)
filtered = data[mask]

# ─── KPIs ────────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("Lignes filtrées",        len(filtered))
k2.metric("Images distinctes",      filtered["image_name"].nunique() if "image_name" in filtered.columns else 0)
k3.metric("Confidence moyenne",     round(float(filtered["confidence_score"].mean()), 3) if len(filtered) else 0)
k4.metric("Bbox moyennes / image",  round(float(filtered["num_boxes"].mean()), 2)        if len(filtered) else 0)

# ─── Charts ──────────────────────────────────────────────────────────────────

c1, c2 = st.columns(2)
with c1:
    st.subheader("Prédiction")
    st.bar_chart(filtered["prediction"].value_counts())

    st.subheader("Taille bbox")
    if "bbox_size_category" in filtered.columns:
        st.bar_chart(filtered["bbox_size_category"].value_counts())

with c2:
    st.subheader("Quality flag")
    st.bar_chart(filtered["quality_flag"].value_counts())

    st.subheader("Split")
    st.bar_chart(filtered["split"].value_counts())

st.subheader("Distribution confidence_score")
st.bar_chart(filtered["confidence_score"].round(1).value_counts().sort_index())

# ─── Table ───────────────────────────────────────────────────────────────────

st.subheader(f"Données enrichies — {len(filtered)} lignes")
st.dataframe(filtered, width="stretch")

# ─── Prédictions dédiées ─────────────────────────────────────────────────────

st.subheader("Prédictions (output/predictions/)")
try:
    preds_df = load_predictions()
    if preds_df.empty:
        st.info("Aucune prédiction disponible.")
    else:
        preds_filtered = preds_df[preds_df["confidence_score"] >= min_conf]
        st.dataframe(preds_filtered, width="stretch")
except Exception as exc:
    st.warning(f"Impossible de lire predictions/ : {exc}")
