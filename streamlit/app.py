"""
Dashboard Streamlit — LECTURE SEULE, en continu (READ 8).

Deux onglets, conformes au cahier des charges :
  1. Suivi des prédictions & du flux : Pipeline Tracker de bout en bout
     (SOURCE → DEST → OUTPUT), table de traçabilité par fichier avec statut
     (Reçu / En cours / Traité), prédictions et score de confiance.
  2. Suivi de l'entraînement : modèle actif (version, date, volume dataset),
     métriques du dernier run hebdo (accuracy, F1, matrice de confusion
     test/val) et évolution historique de la précision.

Les octets images (image_data) ne sont jamais chargés depuis les Parquet ;
l'aperçu ouvre les fichiers images directement sur disque via image_path.
"""

import glob
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SOURCE_INGRESS = os.path.join(_BASE, "source", "continuous_ingress")
DEST_INGRESS   = os.path.join(_BASE, "dest", "continuous_ingress")
OUTPUT_PATH    = os.path.join(_BASE, "output")
METRICS_PATH   = os.path.join(_BASE, "models", "metrics")
MODELS_PATH    = os.path.join(_BASE, "models")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Colonnes lues depuis OUTPUT/ — image_data (binaire lourd) volontairement exclue.
# Le flux continu ne transporte AUCUN label/bbox : le schéma est minimal.
READ_COLUMNS = [
    "image_name", "image_path", "length",
    "ingested_at", "processed_at",
    "prediction", "confidence_score",
]

PRED_COLORS = {"plate": "#2ecc71", "background": "#95a5a6", "uncertain": "#f39c12"}
STATUS_ORDER = ["Reçu", "En cours", "Traité"]


# ─── Chargement des données ──────────────────────────────────────────────────

def _image_files(directory: str):
    if not os.path.isdir(directory):
        return []
    return [p for p in glob.glob(os.path.join(directory, "**", "*"), recursive=True)
            if p.lower().endswith(IMAGE_EXTS)]


@st.cache_data(ttl=30, show_spinner="Lecture des Parquet…")
def load_output() -> pd.DataFrame:
    files = glob.glob(os.path.join(OUTPUT_PATH, "**", "*.parquet"), recursive=True)
    if not files:
        return pd.DataFrame(columns=READ_COLUMNS)
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


@st.cache_data(ttl=30, show_spinner=False)
def load_metrics_history() -> list:
    """Runs hebdo triés du plus ancien au plus récent (MODEL/metrics/*.json)."""
    runs = []
    for path in glob.glob(os.path.join(METRICS_PATH, "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(runs, key=lambda r: r.get("trained_at", ""))


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
def load_image(path: str):
    """Charge une image du disque pour l'aperçu (pas de bbox : le flux
    d'inférence ne transporte aucun label — la prédiction est au niveau image)."""
    from PIL import Image

    with Image.open(path) as img:
        return img.convert("RGB").copy()


def build_traceability(output_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lineage par fichier :
      Reçu     = image encore dans SOURCE/continuous_ingress (en attente Producer)
      En cours = image dans DEST/continuous_ingress, pas encore dans OUTPUT
      Traité   = image présente dans OUTPUT avec prédiction + score
    """
    rows = []
    processed = set(output_df["image_name"].dropna()) if not output_df.empty else set()

    for path in _image_files(SOURCE_INGRESS):
        rows.append({"fichier": os.path.basename(path), "statut": "Reçu",
                     "timestamp": datetime.fromtimestamp(os.path.getmtime(path)),
                     "prediction": None, "confidence_score": None})

    for path in _image_files(DEST_INGRESS):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in processed:
            continue  # apparaîtra en "Traité" via OUTPUT
        rows.append({"fichier": os.path.basename(path), "statut": "En cours",
                     "timestamp": datetime.fromtimestamp(os.path.getmtime(path)),
                     "prediction": None, "confidence_score": None})

    if not output_df.empty:
        done = (output_df.sort_values("processed_at")
                .drop_duplicates("image_name", keep="last"))
        for _, r in done.iterrows():
            rows.append({"fichier": str(r["image_name"]), "statut": "Traité",
                         "timestamp": pd.to_datetime(r["processed_at"]),
                         "prediction": r["prediction"],
                         "confidence_score": r["confidence_score"]})

    if not rows:
        return pd.DataFrame(columns=["fichier", "statut", "timestamp",
                                     "prediction", "confidence_score"])
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

# ═══ Onglet 1 — Pipeline Tracker + traçabilité ═══════════════════════════════

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
        c4.metric("⏱️ Latence moyenne ingestion→prédiction",
                  f"{latency.mean():.1f}s",
                  help="Écart moyen entre le dépôt dans DEST et l'écriture du score.")
    else:
        c4.metric("⏱️ Latence moyenne ingestion→prédiction", "—")

    # Barre de progression du flux
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
        f1, f2 = st.columns([1, 3])
        with f1:
            statut_filter = st.multiselect("Statut", STATUS_ORDER, default=STATUS_ORDER)
        shown = trace[trace["statut"].isin(statut_filter)].head(300)
        st.dataframe(
            shown, width="stretch", hide_index=True,
            column_config={
                "fichier": st.column_config.TextColumn("Fichier"),
                "statut": st.column_config.TextColumn("Statut"),
                "timestamp": st.column_config.DatetimeColumn("Horodatage",
                                                             format="DD/MM HH:mm:ss"),
                "prediction": st.column_config.TextColumn("Prédiction"),
                "confidence_score": st.column_config.ProgressColumn(
                    "Confidence", format="%.2f", min_value=0.0, max_value=1.0),
            },
        )

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

        with st.expander("🖼️ Aperçu des dernières images prédites"):
            recent = (output_df.dropna(subset=["image_path"])
                      .sort_values("processed_at", ascending=False)
                      .drop_duplicates("image_name").head(8))
            cols, shown_imgs = st.columns(4), 0
            for _, row in recent.iterrows():
                local = uri_to_path(row["image_path"])
                if not local:
                    continue
                with cols[shown_imgs % 4]:
                    try:
                        st.image(load_image(local), width="stretch")
                    except Exception:
                        continue
                    badge = PRED_COLORS.get(row["prediction"], "#95a5a6")
                    st.markdown(
                        f"<code>{row['image_name']}</code><br>"
                        f"<span style='color:{badge};font-weight:600'>{row['prediction']}</span>"
                        f" · conf {row['confidence_score']:.2f}",
                        unsafe_allow_html=True)
                shown_imgs += 1

# ═══ Onglet 2 — Suivi de l'entraînement ══════════════════════════════════════

with tab_training:
    runs = load_metrics_history()

    if not runs:
        st.info("**Aucun run d'entraînement enregistré.**\n\n"
                "Lancer le flux hebdomadaire : `python main.py --weekly` — "
                "le Producer copiera `SOURCE/ml_dataset/` vers `DEST/ml_dataset/`, "
                "puis le service ML entraînera le modèle et déposera ses métriques ici.")
    else:
        last = runs[-1]
        trained_at = pd.to_datetime(last.get("trained_at"))

        st.subheader("Modèle actif")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Version", last.get("version", "—"))
        m2.metric("Dernier run hebdo", f"{trained_at:%d/%m/%Y %H:%M}" if trained_at else "—")
        m3.metric("Volume dataset (images)", f"{last.get('dataset_images', 0):,}")
        m4.metric("Runs enregistrés", len(runs))

        pkl_ok = os.path.exists(os.path.join(MODELS_PATH, "plate_model.pkl"))
        onnx_ok = os.path.exists(os.path.join(MODELS_PATH, "plate_model.onnx"))
        st.caption(
            f"Artefacts : plate_model.pkl {'✅' if pkl_ok else '❌'} · "
            f"plate_model.onnx {'✅' if onnx_ok else '❌'} · "
            f"répartition : {last.get('split_counts', {})}")

        st.subheader("Performance du dernier entraînement")
        metrics = last.get("metrics", {})
        if not metrics:
            st.warning("Run enregistré sans métriques (sets test/val vides ?).")
        else:
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

        st.subheader("Évolution historique de la précision")
        history = [
            {"version": r["version"], "trained_at": r["trained_at"],
             "split": split, "accuracy": m["accuracy"], "f1": m["f1_score"]}
            for r in runs for split, m in r.get("metrics", {}).items()
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
                    y=alt.Y("accuracy:Q", title="Accuracy",
                            scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("split:N", title="Set"),
                    tooltip=["version", "split", "accuracy", "f1"],
                ).properties(height=260),
                width="stretch")
