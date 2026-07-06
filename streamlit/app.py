"""
Dashboard Streamlit — LECTURE SEULE, en continu (READ 8).

Lit les Parquet de OUTPUT/ (écrits par ConsumerApp) via pandas + pyarrow,
et se rafraîchit automatiquement pour suivre le stream en temps réel.
Les octets images (image_data) ne sont jamais chargés depuis les Parquet ;
la galerie ouvre les fichiers images directement sur disque via image_path.
"""

import glob
import os
import urllib.parse
import urllib.request

import altair as alt
import pandas as pd
import streamlit as st

_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH = os.path.join(_BASE, "output")

# Colonnes lues depuis OUTPUT/ — image_data (binaire lourd) volontairement exclue
READ_COLUMNS = [
    "image_name", "split", "image_path",
    "label_path", "label_data",
    "class_id", "cx_norm", "cy_norm", "w_norm", "h_norm",
    "bbox_area_norm", "aspect_ratio",
    "prediction", "confidence_score", "processed_at",
]

TABLE_COLUMNS = [
    "image_name", "split", "prediction", "confidence_score",
    "bbox_area_norm", "aspect_ratio", "processed_at",
]

PRED_COLORS = {"plate": "#2ecc71", "background": "#95a5a6", "uncertain": "#f39c12"}


def _parquet_files(directory: str):
    return glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True)


@st.cache_data(ttl=30, show_spinner="Lecture des Parquet…")
def load_output() -> pd.DataFrame:
    files = _parquet_files(OUTPUT_PATH)
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, columns=READ_COLUMNS))
        except Exception:
            # Fichier d'un ancien schéma : on lit tout puis on filtre
            df = pd.read_parquet(f)
            frames.append(df[[c for c in READ_COLUMNS if c in df.columns]])
    # reindex : garantit toutes les colonnes attendues (NaN si schéma ancien)
    return pd.concat(frames, ignore_index=True).reindex(columns=READ_COLUMNS)


def uri_to_path(uri: str):
    """Convertit une URI Spark (file:/C:/...) en chemin local, None si distant."""
    if not uri:
        return None
    if uri.startswith("file:"):
        parsed = urllib.parse.urlparse(uri)
        path = urllib.request.url2pathname(parsed.path)
        return path if os.path.exists(path) else None
    return uri if os.path.exists(uri) else None


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def load_image_with_boxes(path: str, boxes: tuple):
    """Charge une image du disque et dessine les bbox YOLO (normalisées)."""
    from PIL import Image, ImageDraw

    img = Image.open(path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for cx, cy, bw, bh in boxes:
        x0 = (cx - bw / 2) * w
        y0 = (cy - bh / 2) * h
        x1 = (cx + bw / 2) * w
        y1 = (cy + bh / 2) * h
        draw.rectangle([x0, y0, x1, y1], outline="#2ecc71", width=max(2, int(w / 200)))
    return img


def pred_chart(series: pd.Series) -> alt.Chart:
    counts = series.value_counts().rename_axis("prediction").reset_index(name="count")
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("count:Q", title="Nombre de lignes"),
            y=alt.Y("prediction:N", sort="-x", title=None),
            color=alt.Color(
                "prediction:N", legend=None,
                scale=alt.Scale(domain=list(PRED_COLORS), range=list(PRED_COLORS.values())),
            ),
            tooltip=["prediction", "count"],
        )
        .properties(height=180)
    )


def histogram(df: pd.DataFrame, col: str, title: str, maxbins: int = 30) -> alt.Chart:
    return (
        alt.Chart(df[[col]].dropna())
        .mark_bar(color="#4c78a8", cornerRadiusEnd=2)
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=maxbins), title=title),
            y=alt.Y("count()", title="Lignes"),
            tooltip=["count()"],
        )
        .properties(height=220)
    )


# ─── Page ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Plate Detection Dashboard",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚘 Plate Detection — Dashboard temps réel")

# ─── Sidebar : rafraîchissement + filtres ────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Contrôles")
    auto_refresh = st.toggle("Auto-refresh", value=True,
                             help="Relit data/output/ périodiquement sans bloquer l'interface.")
    refresh_seconds = st.slider("Intervalle (secondes)", 5, 120, 30, 5,
                                disabled=not auto_refresh)
    if st.button("🔄 Rafraîchir maintenant", width="stretch"):
        st.cache_data.clear()
        st.rerun()

try:
    data = load_output()
except Exception as exc:
    st.error(f"Impossible de lire `{OUTPUT_PATH}` : {exc}")
    st.stop()

# Rafraîchissement non bloquant : le fragment se réveille tout seul et
# déclenche un rerun global — l'UI reste interactive entre deux ticks.
# Le flag évite que le rendu initial du fragment (dans ce même run) ne
# relance immédiatement le script en boucle.
if auto_refresh:
    st.session_state["_tick_armed"] = False

    @st.fragment(run_every=refresh_seconds)
    def _tick():
        st.caption(f"⏱️ Auto-refresh actif — toutes les {refresh_seconds}s")
        if st.session_state.get("_tick_armed"):
            load_output.clear()
            st.rerun(scope="app")
        st.session_state["_tick_armed"] = True

    _tick()

if data.empty:
    st.info(
        "**Aucune donnée dans `data/output/` pour le moment.**\n\n"
        "1. Lancer le Producer : `python main.py` (ou `--skip-build`)\n"
        "2. Le Consumer écrit les premiers Parquet après ~10 s de streaming\n"
        "3. Ce dashboard se mettra à jour automatiquement"
    )
    st.stop()

with st.sidebar:
    st.divider()
    st.header("🔍 Filtres")
    splits = sorted(data["split"].dropna().unique().tolist())
    predictions = sorted(data["prediction"].dropna().unique().tolist())

    selected_splits = st.multiselect("Split", splits, default=splits)
    selected_preds = st.multiselect("Prédiction", predictions, default=predictions)
    min_conf = st.slider("Confidence minimum", 0.0, 1.0, 0.0, 0.05)
    has_label = st.checkbox("Images avec label seulement", value=False)

    st.divider()
    st.caption(f"📂 Source : `{OUTPUT_PATH}`")

mask = (
    data["split"].isin(selected_splits)
    & data["prediction"].isin(selected_preds)
    & (data["confidence_score"] >= min_conf)
)
if has_label:
    mask &= data["class_id"].notna()

filtered = data[mask]
labeled = filtered[filtered["class_id"].notna()]

if filtered.empty:
    st.warning("Aucune ligne ne correspond aux filtres. Élargir la sélection dans la sidebar.")
    st.stop()

# ─── Bandeau statut ──────────────────────────────────────────────────────────

last_ts = pd.to_datetime(data["processed_at"]).max()
st.caption(
    f"**{len(data):,}** lignes en base · **{data['image_name'].nunique():,}** images · "
    f"dernier traitement : **{last_ts:%d/%m/%Y %H:%M:%S}**"
)

# ─── KPIs ────────────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("Lignes filtrées (1/bbox)", f"{len(filtered):,}")
k2.metric("Images distinctes", f"{filtered['image_name'].nunique():,}")
k3.metric("Confidence moyenne",
          f"{filtered['confidence_score'].mean():.3f}" if len(filtered) else "—")
k4.metric("Bbox YOLO (ground truth)", f"{int(filtered['class_id'].notna().sum()):,}")

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_overview, tab_bbox, tab_gallery, tab_data = st.tabs(
    ["📊 Vue d'ensemble", "📐 Analyse bbox", "🖼️ Galerie", "🗃️ Données"])

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Prédictions")
        st.altair_chart(pred_chart(filtered["prediction"]), width="stretch")
    with c2:
        st.subheader("Répartition par split")
        split_counts = filtered["split"].value_counts().rename_axis("split").reset_index(name="count")
        st.altair_chart(
            alt.Chart(split_counts)
            .mark_bar(cornerRadiusEnd=4, color="#4c78a8")
            .encode(
                x=alt.X("count:Q", title="Nombre de lignes"),
                y=alt.Y("split:N", sort="-x", title=None),
                tooltip=["split", "count"],
            )
            .properties(height=180),
            width="stretch",
        )

    st.subheader("Distribution des confidence scores")
    st.altair_chart(histogram(filtered, "confidence_score", "Confidence score", maxbins=20),
                    width="stretch")

    st.subheader("Débit du stream (images traitées / minute)")
    throughput = (
        filtered.assign(minute=pd.to_datetime(filtered["processed_at"]).dt.floor("min"))
        .groupby("minute")["image_name"].nunique()
        .rename("images").reset_index()
    )
    st.altair_chart(
        alt.Chart(throughput)
        .mark_area(color="#4c78a8", opacity=0.6, line={"color": "#4c78a8"})
        .encode(
            x=alt.X("minute:T", title="Heure de traitement"),
            y=alt.Y("images:Q", title="Images / minute"),
            tooltip=[alt.Tooltip("minute:T", format="%H:%M"), "images"],
        )
        .properties(height=200),
        width="stretch",
    )

with tab_bbox:
    if labeled.empty:
        st.info("Aucune bbox ground truth dans la sélection courante.")
    else:
        b1, b2, b3 = st.columns(3)
        b1.metric("Total bbox", f"{len(labeled):,}")
        b2.metric("Aire moyenne (norm.)", f"{labeled['bbox_area_norm'].mean():.4f}")
        b3.metric("Ratio largeur/hauteur moyen", f"{labeled['aspect_ratio'].mean():.2f}")

        h1, h2 = st.columns(2)
        with h1:
            st.subheader("Aire des bbox")
            st.altair_chart(histogram(labeled, "bbox_area_norm", "Aire normalisée"),
                            width="stretch")
        with h2:
            st.subheader("Aspect ratio")
            st.altair_chart(histogram(labeled, "aspect_ratio", "Largeur / hauteur"),
                            width="stretch")

        st.subheader("Position des centres de bbox dans l'image")
        st.caption("Chaque point est une plaque ; (0,0) = coin haut-gauche de l'image.")
        st.altair_chart(
            alt.Chart(labeled[["cx_norm", "cy_norm", "bbox_area_norm"]].dropna())
            .mark_circle(opacity=0.35, color="#e45756")
            .encode(
                x=alt.X("cx_norm:Q", title="Centre X (normalisé)",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("cy_norm:Q", title="Centre Y (normalisé)",
                        scale=alt.Scale(domain=[0, 1], reverse=True)),
                size=alt.Size("bbox_area_norm:Q", legend=None),
                tooltip=[alt.Tooltip("cx_norm:Q", format=".2f"),
                         alt.Tooltip("cy_norm:Q", format=".2f")],
            )
            .properties(height=380),
            width="stretch",
        )

with tab_gallery:
    st.caption("Images lues directement sur disque via `image_path` — "
               "bbox ground truth dessinées en vert.")
    n_images = st.slider("Nombre d'images affichées", 4, 24, 8, 4)

    with_boxes = filtered.dropna(subset=["cx_norm"])
    boxes_by_image = {
        name: tuple(zip(g["cx_norm"], g["cy_norm"], g["w_norm"], g["h_norm"]))
        for name, g in with_boxes.groupby("image_name")
    }
    gallery = (
        filtered.dropna(subset=["image_path"])
        .drop_duplicates("image_name")
        .head(n_images)
    )

    if gallery.empty:
        st.info("Aucune image disponible dans la sélection courante.")
    else:
        cols = st.columns(4)
        shown = 0
        for _, row in gallery.iterrows():
            local = uri_to_path(row["image_path"])
            if not local:
                continue
            boxes = boxes_by_image.get(row["image_name"], ())
            with cols[shown % 4]:
                try:
                    img = load_image_with_boxes(local, boxes)
                except Exception:
                    continue
                badge = PRED_COLORS.get(row["prediction"], "#95a5a6")
                st.image(img, width="stretch")
                st.markdown(
                    f"<div style='margin-top:-8px'><code>{row['image_name']}</code><br>"
                    f"<span style='color:{badge};font-weight:600'>{row['prediction']}</span>"
                    f" · conf {row['confidence_score']:.2f}"
                    f" · {len(boxes)} bbox</div>",
                    unsafe_allow_html=True,
                )
            shown += 1
        if shown == 0:
            st.warning("Chemins images introuvables sur ce disque "
                       "(dataset déplacé ou dashboard lancé sur une autre machine).")

with tab_data:
    st.subheader(f"Données enrichies — {len(filtered):,} lignes")
    st.dataframe(
        filtered[[c for c in TABLE_COLUMNS if c in filtered.columns]],
        width="stretch",
        hide_index=True,
        column_config={
            "image_name": st.column_config.TextColumn("Image"),
            "split": st.column_config.TextColumn("Split"),
            "prediction": st.column_config.TextColumn("Prédiction"),
            "confidence_score": st.column_config.ProgressColumn(
                "Confidence", format="%.2f", min_value=0.0, max_value=1.0),
            "bbox_area_norm": st.column_config.NumberColumn("Aire bbox", format="%.4f"),
            "aspect_ratio": st.column_config.NumberColumn("Aspect ratio", format="%.2f"),
            "processed_at": st.column_config.DatetimeColumn("Traité le", format="DD/MM HH:mm:ss"),
        },
    )
    st.download_button(
        "⬇️ Exporter la sélection (CSV)",
        filtered[[c for c in TABLE_COLUMNS if c in filtered.columns]].to_csv(index=False),
        file_name="plate_detection_export.csv",
        mime="text/csv",
    )

    with st.expander("Détail labels YOLO (texte brut)"):
        sample = filtered[filtered["label_data"].notna()][["image_name", "label_data"]].head(20)
        st.dataframe(sample, width="stretch", hide_index=True)
