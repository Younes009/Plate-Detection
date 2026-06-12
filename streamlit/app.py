"""
Dashboard Streamlit - LECTURE SEULE.

Lit exclusivement hdfs:///project/processed/ et hdfs:///project/processed/predictions/
(parquet écrits par ConsumerApp). Aucun calcul ML, aucune transformation lourde :
uniquement KPIs, charts, filtres, tables.
"""

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession

PROCESSED_PATH = "hdfs:///project/processed/"
PREDICTIONS_PATH = "hdfs:///project/processed/predictions/"

# Colonnes affichées (on exclut image_data : binaire, inutile pour la visu)
DISPLAY_COLUMNS = [
    "image_name",
    "split",
    "label",
    "class_id",
    "num_boxes",
    "bbox_area_norm",
    "aspect_ratio",
    "bbox_size_category",
    "center_x_quadrant",
    "center_y_quadrant",
    "quality_flag",
    "prediction",
    "confidence_score",
    "processed_at",
]

PREDICTIONS_COLUMNS = ["image_path", "prediction", "confidence_score"]


@st.cache_resource
def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("PlateDetection-Dashboard")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


@st.cache_data(ttl=60)
def load_processed() -> pd.DataFrame:
    spark = get_spark()
    df = spark.read.parquet(PROCESSED_PATH)
    cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    return df.select(*cols).toPandas()


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    spark = get_spark()
    df = spark.read.parquet(PREDICTIONS_PATH)
    cols = [c for c in PREDICTIONS_COLUMNS if c in df.columns]
    return df.select(*cols).toPandas()


st.set_page_config(page_title="Plate Detection - Processed Data", layout="wide")
st.title("Dashboard - hdfs:///project/processed/")
st.caption("Lecture seule (donnees produites par ConsumerApp en streaming)")

if st.button("Rafraichir"):
    st.cache_data.clear()

try:
    data = load_processed()
except Exception as exc:
    st.error(f"Impossible de lire {PROCESSED_PATH} : {exc}")
    st.stop()

if data.empty:
    st.info("Aucune donnee dans processed/ pour le moment.")
    st.stop()

# --- Filtres ---
st.sidebar.header("Filtres")
splits = sorted(data["split"].dropna().unique().tolist())
labels = sorted(data["label"].dropna().unique().tolist())
quality_flags = sorted(data["quality_flag"].dropna().unique().tolist())
min_confidence = st.sidebar.slider("Confidence score minimum", 0.0, 1.0, 0.0, 0.05)

selected_splits = st.sidebar.multiselect("Split", splits, default=splits)
selected_labels = st.sidebar.multiselect("Classe (label)", labels, default=labels)
selected_quality = st.sidebar.multiselect("Quality flag", quality_flags, default=quality_flags)

filtered = data[
    data["split"].isin(selected_splits)
    & data["label"].isin(selected_labels)
    & data["quality_flag"].isin(selected_quality)
    & (data["confidence_score"] >= min_confidence)
]

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Plaques detectees", len(filtered))
col2.metric("Images distinctes", filtered["image_name"].nunique())
col3.metric("Confidence moyenne", round(filtered["confidence_score"].mean(), 2) if len(filtered) else 0)
col4.metric("Plaques moyennes / image", round(filtered["num_boxes"].mean(), 2) if len(filtered) else 0)

# --- Charts ---
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Repartition par classe (label)")
    st.bar_chart(filtered["label"].value_counts())

    st.subheader("Repartition par prediction")
    st.bar_chart(filtered["prediction"].value_counts())

with chart_col2:
    st.subheader("Repartition par taille de bbox")
    st.bar_chart(filtered["bbox_size_category"].value_counts())

    st.subheader("Repartition par quality_flag")
    st.bar_chart(filtered["quality_flag"].value_counts())

st.subheader("Distribution des confidence_score")
st.bar_chart(filtered["confidence_score"].round(1).value_counts().sort_index())

# --- Table principale ---
st.subheader("Donnees detaillees (processed/)")
st.dataframe(filtered, use_container_width=True)

# --- Predictions dediees ---
st.subheader("Predictions (processed/predictions/)")
try:
    predictions = load_predictions()
    if predictions.empty:
        st.info("Aucune prediction disponible pour le moment.")
    else:
        st.dataframe(predictions, use_container_width=True)
except Exception as exc:
    st.warning(f"Impossible de lire {PREDICTIONS_PATH} : {exc}")
