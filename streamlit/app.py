"""
Streamlit dashboard — READ-ONLY continuous view of the pipeline.

Two tabs:
  1. Flow & predictions: end-to-end Pipeline Tracker (SOURCE -> DEST ->
     OUTPUT), per-file traceability (Recu / En cours / Traite), predicted
     bbox drawn on recent images.
  2. Training: active model version, accuracy / F1 / confusion matrices,
     locator IoU, accuracy history across weekly runs.

image_data bytes are never loaded from Parquet; previews open files on disk.
"""

import glob
import html
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SOURCE_INGRESS  = os.path.join(_BASE, "source", "continuous_ingress")
DEST_INGRESS    = os.path.join(_BASE, "dest", "continuous_ingress")
ARCHIVE_INGRESS = os.path.join(_BASE, "archive", "continuous_ingress")
OUTPUT_PATH     = os.path.join(_BASE, "output")
METRICS_PATH    = os.path.join(_BASE, "models", "metrics")
MODELS_PATH     = os.path.join(_BASE, "models")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# image_data (heavy binary) intentionally excluded
READ_COLUMNS = [
    "image_name", "image_path", "length",
    "ingested_at", "processed_at",
    "prediction", "confidence_score",
    "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
]

PRED_COLORS = {"plate": "#2ecc71", "background": "#95a5a6", "uncertain": "#f39c12"}
STATUS_ORDER = ["Reçu", "En cours", "Traité"]


# ─── Data loading ────────────────────────────────────────────────────────────

def _image_files(directory: str):
    if not os.path.isdir(directory):
        return []
    return [p for p in glob.glob(os.path.join(directory, "**", "*"), recursive=True)
            if p.lower().endswith(IMAGE_EXTS)]


@st.cache_data(ttl=30, show_spinner="Lecture des Parquet…")
def load_output() -> pd.DataFrame:
    files = glob.glob(os.path.join(OUTPUT_PATH, "**", "*.parquet"), recursive=True)
    frames = []
    for f in files:
        # skip files Spark is still writing: _temporary dirs, empty part files
        try:
            if "_temporary" in f or os.path.getsize(f) == 0:
                continue
        except OSError:
            continue
        try:
            frames.append(pd.read_parquet(f, columns=READ_COLUMNS))
        except Exception:
            try:
                # older schema: read everything then filter
                df = pd.read_parquet(f)
                frames.append(df[[c for c in READ_COLUMNS if c in df.columns]])
            except Exception:
                continue  # incomplete footer (mid-write) — next refresh gets it
    if not frames:
        return pd.DataFrame(columns=READ_COLUMNS)
    # reindex guarantees all expected columns (NaN for old schemas)
    return pd.concat(frames, ignore_index=True).reindex(columns=READ_COLUMNS)


@st.cache_data(ttl=30, show_spinner=False)
def load_metrics_history() -> list:
    """Weekly runs, oldest first (MODEL/metrics/*.json)."""
    runs = []
    for path in glob.glob(os.path.join(METRICS_PATH, "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(runs, key=lambda r: r.get("trained_at", ""))


def uri_to_path(uri: str):
    """Spark URI (file:/C:/...) -> local path. Falls back to the archive:
    the Consumer moves DEST files there after processing."""
    if not uri:
        return None
    path = uri
    if uri.startswith("file:"):
        path = urllib.request.url2pathname(urllib.parse.urlparse(uri).path)
    if os.path.exists(path):
        return path
    archived = os.path.join(ARCHIVE_INGRESS, os.path.basename(path))
    return archived if os.path.exists(archived) else None


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def load_image_with_pred_box(path: str, box):
    """Draw the PREDICTED bbox (model output, normalized YOLO coords)."""
    from PIL import Image, ImageDraw

    with Image.open(path) as raw:
        img = raw.convert("RGB").copy()
    if box is not None:
        w, h = img.size
        cx, cy, bw, bh = box
        draw = ImageDraw.Draw(img)
        draw.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h],
                       outline="#2ecc71", width=max(2, int(w / 150)))
    return img


TRACE_COLUMNS = ["fichier", "statut", "timestamp", "prediction",
                 "confidence_score", "image_path", "bbox"]


def build_traceability(output_df: pd.DataFrame) -> pd.DataFrame:
    """Per-file lineage: Recu (still in SOURCE), En cours (in DEST, not yet
    in OUTPUT), Traite (in OUTPUT with prediction + score). image_path and
    bbox are kept (hidden in the table) to preview the clicked row."""
    rows = []
    processed = set(output_df["image_name"].dropna()) if not output_df.empty else set()

    for path in _image_files(SOURCE_INGRESS):
        rows.append({"fichier": os.path.basename(path), "statut": "Reçu",
                     "timestamp": datetime.fromtimestamp(os.path.getmtime(path)),
                     "prediction": None, "confidence_score": None,
                     "image_path": path, "bbox": None})

    for path in _image_files(DEST_INGRESS):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in processed:
            continue
        rows.append({"fichier": os.path.basename(path), "statut": "En cours",
                     "timestamp": datetime.fromtimestamp(os.path.getmtime(path)),
                     "prediction": None, "confidence_score": None,
                     "image_path": path, "bbox": None})

    if not output_df.empty:
        done = (output_df.sort_values("processed_at")
                .drop_duplicates("image_name", keep="last"))
        for _, r in done.iterrows():
            bbox = None
            if pd.notna(r.get("bbox_cx")):
                bbox = (r["bbox_cx"], r["bbox_cy"], r["bbox_w"], r["bbox_h"])
            rows.append({"fichier": str(r["image_name"]), "statut": "Traité",
                         "timestamp": pd.to_datetime(r["processed_at"]),
                         "prediction": r["prediction"],
                         "confidence_score": r["confidence_score"],
                         "image_path": r["image_path"], "bbox": bbox})

    if not rows:
        return pd.DataFrame(columns=TRACE_COLUMNS)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp", ascending=False)


def confusion_chart(matrix: list, title: str) -> alt.Chart:
    labels = ["background", "plate"]
    records = [{"réel": labels[i], "prédit": labels[j], "n": matrix[i][j]}
               for i in range(2) for j in range(2)]
    base = alt.Chart(pd.DataFrame(records)).properties(height=220, title=title)
    heat = base.mark_rect().encode(
        x=alt.X("prédit:N", title="Prédit"),
        y=alt.Y("réel:N", title="Réel"),
        color=alt.Color("n:Q", scale=alt.Scale(scheme="greens"), legend=None),
    )
    text = base.mark_text(fontSize=18, fontWeight="bold").encode(
        x="prédit:N", y="réel:N", text="n:Q",
        color=alt.condition("datum.n > 100", alt.value("white"), alt.value("#2ecc71")),
    )
    return heat + text


# ─── Page ────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Plate Detection Dashboard",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚘 Plate Detection — Pipeline temps réel")

with st.sidebar:
    st.header("⚙️ Contrôles")
    auto_refresh = st.toggle("Auto-refresh", value=True,
                             help="Relit le pipeline périodiquement sans bloquer l'interface.")
    refresh_seconds = st.slider("Intervalle (secondes)", 5, 120, 30, 5,
                                disabled=not auto_refresh)
    if st.button("🔄 Rafraîchir maintenant", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption(f"📂 SOURCE : `{SOURCE_INGRESS}`")
    st.caption(f"📂 DEST : `{DEST_INGRESS}`")
    st.caption(f"📂 OUTPUT : `{OUTPUT_PATH}`")

# Non-blocking refresh: the fragment wakes up alone and triggers a full
# rerun. The flag stops the fragment's initial render (same run) from
# relaunching the script in a loop.
if auto_refresh:
    st.session_state["_tick_armed"] = False

    @st.fragment(run_every=refresh_seconds)
    def _tick():
        st.caption(f"⏱️ Auto-refresh actif — toutes les {refresh_seconds}s")
        if st.session_state.get("_tick_armed"):
            st.cache_data.clear()
            st.rerun(scope="app")
        st.session_state["_tick_armed"] = True

    _tick()

try:
    output_df = load_output()
except Exception as exc:
    st.error(f"Impossible de lire `{OUTPUT_PATH}` : {exc}")
    st.stop()

tab_flux, tab_training = st.tabs(
    ["📡 Suivi des prédictions & du flux", "🧠 Suivi de l'entraînement"])

# ═══ Tab 1 — Pipeline Tracker + traceability ═════════════════════════════════

with tab_flux:
    st.subheader("Pipeline Tracker — lineage de bout en bout")

    n_source = len(_image_files(SOURCE_INGRESS))
    n_dest_total = len(_image_files(DEST_INGRESS))
    n_done = output_df["image_name"].nunique() if not output_df.empty else 0
    n_transit = max(0, n_dest_total - n_done)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📥 En attente (SOURCE)", f"{n_source:,}",
              help="Images déposées dans SOURCE/continuous_ingress, pas encore ingérées.")
    c2.metric("🚚 En transit (DEST)", f"{n_transit:,}",
              help="Images déplacées par le Producer, pas encore traitées par le Consumer.")
    c3.metric("✅ Traités (OUTPUT)", f"{n_done:,}",
              help="Images passées par tout le pipeline, prédiction disponible.")
    if n_done:
        latency = (pd.to_datetime(output_df["processed_at"])
                   - pd.to_datetime(output_df["ingested_at"])).dt.total_seconds()
        c4.metric("⏱️ Latence moyenne ingestion→prédiction", f"{latency.mean():.1f}s",
                  help="Écart moyen entre le dépôt dans DEST et l'écriture du score.")
    else:
        c4.metric("⏱️ Latence moyenne ingestion→prédiction", "—")

    total_seen = n_source + n_transit + n_done
    if total_seen:
        st.progress(n_done / total_seen,
                    text=f"{n_done}/{total_seen} fichiers arrivés au bout du pipeline")

    st.subheader("Traçabilité des fichiers")
    trace = build_traceability(output_df)
    if trace.empty:
        st.info("**Aucun fichier vu par le pipeline pour le moment.**\n\n"
                "Déposer des images dans `SOURCE/continuous_ingress/` "
                "(ou `python main.py --seed 50`), le Producer les déplacera "
                "et le Consumer les traitera automatiquement.")
    else:
        statut_filter = st.multiselect("Statut", STATUS_ORDER, default=STATUS_ORDER)
        shown = (trace[trace["statut"].isin(statut_filter)]
                 .head(300).reset_index(drop=True))

        t_table, t_preview = st.columns([5, 3])
        with t_table:
            # single-row selection: clicking a row previews the image
            event = st.dataframe(
                shown, width="stretch", hide_index=True,
                on_select="rerun", selection_mode="single-row",
                key="trace_table",
                column_config={
                    "fichier": st.column_config.TextColumn("Fichier"),
                    "statut": st.column_config.TextColumn("Statut"),
                    "timestamp": st.column_config.DatetimeColumn("Horodatage",
                                                                 format="DD/MM HH:mm:ss"),
                    "prediction": st.column_config.TextColumn("Prédiction"),
                    "confidence_score": st.column_config.ProgressColumn(
                        "Confidence", format="%.2f", min_value=0.0, max_value=1.0),
                    "image_path": None,  # hidden, used by the preview
                    "bbox": None,
                },
            )
        with t_preview:
            selected = event.selection.rows if event and event.selection else []
            if not selected:
                st.info("Cliquer une ligne du tableau pour afficher l'image "
                        "(bbox prédite dessinée si l'image est traitée).")
            else:
                row = shown.iloc[selected[0]]
                local = uri_to_path(row["image_path"])
                if local is None:
                    st.warning(f"`{row['fichier']}` introuvable sur disque "
                               "(déplacée ou purgée).")
                else:
                    st.image(load_image_with_pred_box(local, row["bbox"]),
                             width="stretch")
                    if row["statut"] == "Traité":
                        badge = PRED_COLORS.get(row["prediction"], "#95a5a6")
                        st.markdown(
                            f"<code>{html.escape(str(row['fichier']))}</code> · "
                            f"<span style='color:{badge};font-weight:600'>"
                            f"{html.escape(str(row['prediction']))}</span>"
                            f" · conf {row['confidence_score']:.2f}"
                            + (" · 🎯 bbox prédite" if row["bbox"] else ""),
                            unsafe_allow_html=True)
                    else:
                        st.caption(f"`{row['fichier']}` — statut {row['statut']}, "
                                   "pas encore prédite.")

    if not output_df.empty:
        st.subheader("Prédictions du flux")
        g1, g2 = st.columns(2)
        with g1:
            counts = (output_df["prediction"].value_counts()
                      .rename_axis("prediction").reset_index(name="n"))
            st.altair_chart(
                alt.Chart(counts).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X("n:Q", title="Lignes"),
                    y=alt.Y("prediction:N", sort="-x", title=None),
                    color=alt.Color("prediction:N", legend=None,
                                    scale=alt.Scale(domain=list(PRED_COLORS),
                                                    range=list(PRED_COLORS.values()))),
                    tooltip=["prediction", "n"],
                ).properties(height=180),
                width="stretch")
        with g2:
            st.altair_chart(
                alt.Chart(output_df[["confidence_score"]].dropna())
                .mark_bar(color="#4c78a8", cornerRadiusEnd=2)
                .encode(
                    x=alt.X("confidence_score:Q", bin=alt.Bin(maxbins=20),
                            title="Confidence score"),
                    y=alt.Y("count()", title="Lignes"),
                ).properties(height=180),
                width="stretch")

        throughput = (
            output_df.assign(minute=pd.to_datetime(output_df["processed_at"]).dt.floor("min"))
            .groupby("minute")["image_name"].nunique().rename("images").reset_index())
        st.altair_chart(
            alt.Chart(throughput)
            .mark_area(color="#4c78a8", opacity=0.6, line={"color": "#4c78a8"})
            .encode(x=alt.X("minute:T", title="Heure de traitement"),
                    y=alt.Y("images:Q", title="Images / minute"),
                    tooltip=[alt.Tooltip("minute:T", format="%H:%M"), "images"])
            .properties(height=180, title="Débit du stream"),
            width="stretch")

        st.subheader("🖼️ Détections — bbox prédite autour de la plaque")
        # slider bounded by the number of processed images (perf cap at 100)
        max_show = int(max(0, n_done))
        if max_show > 4:
            n_show = st.slider("Nombre d'images affichées", 4, max_show,
                               min(8, max_show),
                               help=f"{n_done:,} images traitées disponibles.")
        else:
            n_show = max_show
        recent = (output_df.dropna(subset=["image_path"])
                  .sort_values("processed_at", ascending=False)
                  .drop_duplicates("image_name").head(n_show))
        cols, shown_imgs = st.columns(4), 0
        for _, row in recent.iterrows():
            local = uri_to_path(row["image_path"])
            if not local:
                continue
            box = None
            if pd.notna(row.get("bbox_cx")):
                box = (row["bbox_cx"], row["bbox_cy"], row["bbox_w"], row["bbox_h"])
            with cols[shown_imgs % 4]:
                try:
                    st.image(load_image_with_pred_box(local, box), width="stretch")
                except Exception:
                    continue
                badge = PRED_COLORS.get(row["prediction"], "#95a5a6")
                bbox_note = "· 🎯 bbox prédite" if box else "· pas de bbox"
                # file names are external data injected into HTML
                st.markdown(
                    f"<code>{html.escape(str(row['image_name']))}</code><br>"
                    f"<span style='color:{badge};font-weight:600'>"
                    f"{html.escape(str(row['prediction']))}</span>"
                    f" · conf {row['confidence_score']:.2f} {bbox_note}",
                    unsafe_allow_html=True)
            shown_imgs += 1

# ═══ Tab 2 — Training monitoring ═════════════════════════════════════════════

with tab_training:
    runs = load_metrics_history()

    if not runs:
        st.info("**Aucun run d'entraînement enregistré.**\n\n"
                "Lancer le flux hebdomadaire : `python main.py --weekly` puis "
                "`python ml/train_yolo.py` — les métriques apparaîtront ici.")
    else:
        last = runs[-1]
        trained_at = pd.to_datetime(last.get("trained_at"))

        st.subheader("Modèle actif")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Version", last.get("version", "—"))
        m2.metric("Dernier run hebdo", f"{trained_at:%d/%m/%Y %H:%M}" if trained_at else "—")
        m3.metric("Volume dataset (images)", f"{last.get('dataset_images', 0):,}")
        m4.metric("Runs enregistrés", len(runs))

        loc_ok = os.path.exists(os.path.join(MODELS_PATH, "plate_locator.onnx"))
        st.caption(
            f"Artefacts : plate_locator.onnx {'✅' if loc_ok else '❌'} · "
            f"modèle : {last.get('model', '?')} · "
            f"répartition : {last.get('split_counts', {})}")

        # classifier metrics only exist for legacy runs; YOLO runs skip this
        metrics = last.get("metrics", {})
        if metrics:
            st.subheader("Performance du classifieur")
            kpi_cols = st.columns(2 * max(1, len(metrics)))
            for i, (split, m) in enumerate(metrics.items()):
                kpi_cols[2 * i].metric(f"Accuracy ({split})", f"{m['accuracy']:.3f}")
                kpi_cols[2 * i + 1].metric(f"F1-score ({split})", f"{m['f1_score']:.3f}")

            cm_cols = st.columns(max(1, len(metrics)))
            for i, (split, m) in enumerate(metrics.items()):
                with cm_cols[i]:
                    st.altair_chart(
                        confusion_chart(m["confusion_matrix"],
                                        f"Matrice de confusion — {split} "
                                        f"(n={m['n_samples']:,})"),
                        width="stretch")

        locator = last.get("locator", {})
        if locator:
            st.subheader("Localisation — qualité des bbox prédites (IoU vs ground truth)")
            loc_cols = st.columns(3 * len(locator))
            for i, (split, m) in enumerate(locator.items()):
                loc_cols[3 * i].metric(f"IoU moyen ({split})", f"{m.get('iou_mean', 0):.3f}")
                loc_cols[3 * i + 1].metric(f"IoU ≥ 0.5 ({split})",
                                           f"{m.get('pct_iou_over_50', 0):.0%}")
                # third metric depends on the locator: MAE (CNN) or median (YOLO)
                if "mae" in m:
                    loc_cols[3 * i + 2].metric(f"MAE coords ({split})", f"{m['mae']:.4f}")
                else:
                    loc_cols[3 * i + 2].metric(f"IoU médian ({split})",
                                               f"{m.get('iou_median', 0):.3f}")
            if any("model" in m for m in locator.values()):
                st.caption(f"Localisateur : {next(iter(locator.values())).get('model', '?')}")

        st.subheader("Évolution historique — IoU du détecteur")
        history = [
            {"version": r["version"], "trained_at": r["trained_at"],
             "split": split, "iou": m.get("iou_mean", 0)}
            for r in runs for split, m in r.get("locator", {}).items()
        ]
        if len(history) < 2:
            st.caption("L'historique se remplira au fil des runs hebdomadaires.")
        if history:
            hist_df = pd.DataFrame(history)
            hist_df["trained_at"] = pd.to_datetime(hist_df["trained_at"])
            st.altair_chart(
                alt.Chart(hist_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("trained_at:T", title="Date du run"),
                    y=alt.Y("iou:Q", title="IoU moyen",
                            scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("split:N", title="Set"),
                    tooltip=["version", "split", "iou"],
                ).properties(height=260),
                width="stretch")
