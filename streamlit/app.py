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
    "image_name", "split", "image_path",
    "prediction", "confidence_score", "processed_at",
    "label_path", "label_data",
]

PREDICTIONS_COLUMNS = [
    "image_path", "prediction", "confidence_score", "processed_at", "label_data",
]


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


def _parse_yolo_boxes(label_data: str) -> list[dict]:
    """Parse YOLO label string → liste de bbox {cx, cy, w, h, area}."""
    if not label_data or pd.isna(label_data):
        return []
    boxes = []
    for line in str(label_data).strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append({"cx": cx, "cy": cy, "w": w, "h": h, "area": w * h})
            except ValueError:
                pass
    return boxes


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
splits      = sorted(data["split"].dropna().unique().tolist()) if "split" in data.columns else []
predictions = sorted(data["prediction"].dropna().unique().tolist()) if "prediction" in data.columns else []

selected_splits = st.sidebar.multiselect("Split",      splits,      default=splits)
selected_preds  = st.sidebar.multiselect("Prédiction", predictions, default=predictions)
min_conf = st.sidebar.slider("Confidence score minimum", 0.0, 1.0, 0.0, 0.05)
has_label = st.sidebar.checkbox("Images avec label seulement", value=False)

mask = (
    data["split"].isin(selected_splits)
    & data["prediction"].isin(selected_preds)
    & (data["confidence_score"] >= min_conf)
)
if has_label and "label_data" in data.columns:
    mask &= data["label_data"].notna()

filtered = data[mask]

# ─── KPIs ────────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("Images filtrées",    len(filtered))
k2.metric("Images distinctes",  filtered["image_name"].nunique() if "image_name" in filtered.columns else 0)
k3.metric("Confidence moyenne", round(float(filtered["confidence_score"].mean()), 3) if len(filtered) else 0)

if "label_data" in filtered.columns:
    labeled = filtered["label_data"].notna().sum()
    k4.metric("Avec label YOLO", int(labeled))

# ─── Charts ──────────────────────────────────────────────────────────────────

c1, c2 = st.columns(2)
with c1:
    st.subheader("Prédiction")
    st.bar_chart(filtered["prediction"].value_counts())

with c2:
    st.subheader("Split")
    st.bar_chart(filtered["split"].value_counts())

st.subheader("Distribution confidence_score")
st.bar_chart(filtered["confidence_score"].round(1).value_counts().sort_index())

# ─── Statistiques bbox (ground truth) ────────────────────────────────────────

if "label_data" in filtered.columns:
    labeled_df = filtered[filtered["label_data"].notna()].copy()
    if not labeled_df.empty:
        st.subheader("Statistiques bbox YOLO (ground truth)")
        all_boxes = []
        for _, row in labeled_df.iterrows():
            for box in _parse_yolo_boxes(row["label_data"]):
                box["image_name"] = row.get("image_name", "")
                all_boxes.append(box)
        if all_boxes:
            bbox_df = pd.DataFrame(all_boxes)
            b1, b2, b3 = st.columns(3)
            b1.metric("Total bbox", len(bbox_df))
            b2.metric("Aire moyenne",  round(float(bbox_df["area"].mean()), 4))
            b3.metric("Max bbox / image",
                      int(bbox_df.groupby("image_name").size().max()) if "image_name" in bbox_df.columns else "—")

            st.bar_chart(
                bbox_df["area"].round(3).value_counts().sort_index(),
            )

# ─── Table ───────────────────────────────────────────────────────────────────

st.subheader(f"Données enrichies — {len(filtered)} lignes")
display_cols = [c for c in DISPLAY_COLUMNS if c in filtered.columns and c not in ("label_data", "image_data")]
st.dataframe(filtered[display_cols], use_container_width=True)

with st.expander("Détail labels YOLO (texte brut)"):
    if "label_data" in filtered.columns:
        sample = filtered[filtered["label_data"].notna()][["image_name", "label_data"]].head(20)
        st.dataframe(sample, use_container_width=True)

# ─── Prédictions dédiées ─────────────────────────────────────────────────────

st.subheader("Prédictions (output/predictions/)")
try:
    preds_df = load_predictions()
    if preds_df.empty:
        st.info("Aucune prédiction disponible.")
    else:
        preds_filtered = preds_df[preds_df["confidence_score"] >= min_conf]
        show_cols = [c for c in PREDICTIONS_COLUMNS if c in preds_filtered.columns and c != "label_data"]
        st.dataframe(preds_filtered[show_cols], use_container_width=True)
except Exception as exc:
    st.warning(f"Impossible de lire predictions/ : {exc}")
