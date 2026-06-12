"""
Dashboard Streamlit - VISUALISATION UNIQUEMENT.

Lit exclusivement hdfs:///project/processed/ (parquet écrit par ConsumerApp).
Aucun calcul métier, aucun scoring ML : uniquement charts, filtres, tables.
"""

import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession

PROCESSED_PATH = "hdfs:///project/processed/"
AGGREGATES_PATH = "hdfs:///project/processed/aggregates/"

# Colonnes affichées (on exclut image_data : binaire, inutile pour la visu)
DISPLAY_COLUMNS = [
    "image_name",
    "split",
    "class_id",
    "num_boxes",
    "bbox_area_norm",
    "aspect_ratio",
    "bbox_size_category",
    "center_x_quadrant",
    "center_y_quadrant",
    "quality_flag",
    "processed_at",
]

AGG_COLUMNS = [
    "window",
    "split",
    "bbox_size_category",
    "nb_images",
    "avg_aspect_ratio",
    "avg_num_boxes",
]


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
def load_aggregates() -> pd.DataFrame:
    spark = get_spark()
    df = spark.read.parquet(AGGREGATES_PATH)
    cols = [c for c in AGG_COLUMNS if c in df.columns]
    pdf = df.select(*cols).toPandas()
    if "window" in pdf.columns:
        pdf["window_start"] = pdf["window"].apply(lambda w: w["start"])
        pdf = pdf.drop(columns=["window"])
    return pdf


st.set_page_config(page_title="Plate Detection - Processed Data", layout="wide")
st.title("Dashboard - hdfs:///project/processed/")
st.caption("Visualisation uniquement (donnees produites par ConsumerApp en streaming)")

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
size_categories = sorted(data["bbox_size_category"].dropna().unique().tolist())
quality_flags = sorted(data["quality_flag"].dropna().unique().tolist())

selected_splits = st.sidebar.multiselect("Split", splits, default=splits)
selected_sizes = st.sidebar.multiselect("Taille bbox", size_categories, default=size_categories)
selected_quality = st.sidebar.multiselect("Quality flag", quality_flags, default=quality_flags)

filtered = data[
    data["split"].isin(selected_splits)
    & data["bbox_size_category"].isin(selected_sizes)
    & data["quality_flag"].isin(selected_quality)
]

# --- KPIs ---
col1, col2, col3 = st.columns(3)
col1.metric("Lignes (filtrees)", len(filtered))
col2.metric("Images distinctes", filtered["image_name"].nunique())
col3.metric("Plaques moyennes / image", round(filtered["num_boxes"].mean(), 2) if len(filtered) else 0)

# --- Charts ---
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Repartition par taille de bbox")
    st.bar_chart(filtered["bbox_size_category"].value_counts())

    st.subheader("Repartition par quality_flag")
    st.bar_chart(filtered["quality_flag"].value_counts())

with chart_col2:
    st.subheader("Quadrant du centre (X)")
    st.bar_chart(filtered["center_x_quadrant"].value_counts())

    st.subheader("Quadrant du centre (Y)")
    st.bar_chart(filtered["center_y_quadrant"].value_counts())

st.subheader("Distribution aspect_ratio")
st.bar_chart(filtered["aspect_ratio"].round(1).value_counts().sort_index())

# --- Table ---
st.subheader("Donnees detaillees")
st.dataframe(filtered, use_container_width=True)

# --- Agregats fenetres ---
st.subheader("Agregats par fenetre (processed/aggregates/)")
try:
    agg = load_aggregates()
    if agg.empty:
        st.info("Aucun agregat disponible pour le moment.")
    else:
        st.dataframe(agg, use_container_width=True)
        st.line_chart(agg.set_index("window_start")["nb_images"])
except Exception as exc:
    st.warning(f"Impossible de lire {AGGREGATES_PATH} : {exc}")
