"""
main.py — Orchestrateur des DEUX temporalités du pipeline :

FLUX CONTINU (défaut) — inférence au fil de l'eau :
  1. Build des fat-jars (skippé si --skip-build)
  2. Producer mode continuous (arrière-plan) :
       SOURCE/continuous_ingress/ → DEST/continuous_ingress/ (déplacement brut)
  3. Consumer Structured Streaming (arrière-plan) :
       DEST/continuous_ingress/ → staging → PREDICT → OUTPUT
  4. Streamlit : dashboard → http://localhost:8501

FLUX HEBDOMADAIRE (--weekly) — réentraînement du modèle :
  1. Producer mode batch :
       SOURCE/ml_dataset/ (train/test/val) → DEST/ml_dataset/ (copie récursive)
  2. ml/train_model.py : lit DEST/ml_dataset/, entraîne, exporte .pkl + .onnx
     dans MODEL/ et versionne les métriques (lues par l'onglet 2 du dashboard)

Usage :
    python main.py                     # flux continu complet
    python main.py --weekly            # run hebdomadaire (Producer batch + ML)
    python main.py --seed 50           # dépose 50 images test dans SOURCE/continuous_ingress
    python main.py --skip-build        # ne pas rebuilder les jars
    python main.py --streamlit-only    # dashboard seulement

Planification hebdo : schtasks (Windows) ou cron (Linux) sur `python main.py --weekly`.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import time

# Console Windows en cp1252 : évite UnicodeEncodeError sur les flèches/emoji
# quand stdout est redirigé (fichier, CI, tâche en arrière-plan).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Chemins ─────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

SOURCE_INGRESS = os.path.join(DATA, "source", "continuous_ingress")
SOURCE_ML      = os.path.join(DATA, "source", "ml_dataset")
DEST_INGRESS   = os.path.join(DATA, "dest", "continuous_ingress")
DEST_ML        = os.path.join(DATA, "dest", "ml_dataset")
STAGING        = os.path.join(DATA, "staging")
OUTPUT         = os.path.join(DATA, "output")
MODELS         = os.path.join(DATA, "models")
CHECKPOINT     = os.path.join(DATA, "checkpoint")

# Si SOURCE/ml_dataset n'existe pas, le dataset livré joue ce rôle
# (évite de dupliquer 10 000 images sur le disque).
DATASET_FALLBACK = os.path.join(BASE, "license-plate-detection-dataset-10125-images")

PRODUCER_JAR = os.path.join(BASE, "producer", "target", "scala-2.13", "producer-assembly-0.1.0.jar")
CONSUMER_JAR = os.path.join(BASE, "consumer", "target", "scala-2.13", "consumer-assembly-0.1.0.jar")
SBT_JAR      = os.path.join(BASE, "sbt-launch.jar")

# shutil.which résout spark-submit.cmd sous Windows (Popen ne le fait pas seul)
SPARK_SUBMIT = shutil.which("spark-submit") or "spark-submit"

SPARK_SUBMIT_ARGS = [
    SPARK_SUBMIT,
    "--master", "local[4]",
    "--driver-memory", "2g",
    "--conf", "spark.driver.extraJavaOptions=-Djava.awt.headless=true",
    # Force le loopback : sinon Spark se bind sur un adaptateur virtuel
    # (VPN/WSL/VMware) et les RPC driver<->executor sont coupés sous Windows.
    "--conf", "spark.driver.bindAddress=127.0.0.1",
    "--conf", "spark.driver.host=localhost",
    "--conf", "spark.network.timeout=300s",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def run(cmd, **kwargs):
    print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def ensure_dirs():
    for d in [SOURCE_INGRESS, DEST_INGRESS, DEST_ML, STAGING, OUTPUT, MODELS, CHECKPOINT]:
        os.makedirs(d, exist_ok=True)


def ml_source_dir() -> str:
    if os.path.isdir(SOURCE_ML):
        return SOURCE_ML
    if os.path.isdir(DATASET_FALLBACK):
        print(f"[INFO] SOURCE/ml_dataset absent — le dataset livré joue ce rôle : {DATASET_FALLBACK}")
        return DATASET_FALLBACK
    print(f"[ERROR] Aucun dataset : ni {SOURCE_ML} ni {DATASET_FALLBACK}")
    sys.exit(1)


def check_jar(jar: str):
    if not os.path.exists(jar):
        print(f"[ERROR] Jar introuvable : {jar}\n  → Relance sans --skip-build")
        sys.exit(1)


# ─── Étapes ──────────────────────────────────────────────────────────────────

def build():
    log("BUILD — fat-jars (sbt assembly)")
    if not os.path.exists(SBT_JAR):
        print(f"[ERROR] sbt-launch.jar introuvable : {SBT_JAR}")
        sys.exit(1)
    run(["java", "-Xmx512m", "-jar", SBT_JAR, "assembly"])
    for jar in [PRODUCER_JAR, CONSUMER_JAR]:
        check_jar(jar)
    print("[OK] Jars construits.")


def seed_ingress(n: int):
    """Simule des arrivées : copie n images (+ labels) du set test vers SOURCE/continuous_ingress."""
    log(f"SEED — {n} images test → {SOURCE_INGRESS}")
    src = os.path.join(ml_source_dir(), "test", "images")
    if not os.path.isdir(src):
        print(f"[ERROR] Pas d'images test dans {src}")
        sys.exit(1)
    images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    picked = random.sample(images, min(n, len(images)))
    for name in picked:
        shutil.copy2(os.path.join(src, name), os.path.join(SOURCE_INGRESS, name))
        label = os.path.join(os.path.dirname(src), "labels",
                             os.path.splitext(name)[0] + ".txt")
        if os.path.exists(label):
            shutil.copy2(label, os.path.join(SOURCE_INGRESS, os.path.basename(label)))
    print(f"[OK] {len(picked)} images déposées dans SOURCE/continuous_ingress.")


def run_producer_batch(batch_size: int, interval: int):
    """Flux HEBDO : copie récursive intégrale de ml_dataset (arborescence préservée)."""
    log("HEBDO 1/2 — Producer batch : SOURCE/ml_dataset → DEST/ml_dataset")
    check_jar(PRODUCER_JAR)
    run(SPARK_SUBMIT_ARGS + [
        "--class", "com.project.producer.ProducerApp", PRODUCER_JAR,
        ml_source_dir(), DEST_ML, str(batch_size), str(interval), "true", "batch",
    ])
    print("[OK] Copie hebdomadaire terminée.")


def run_training():
    log("HEBDO 2/2 — Entraînement ML (DEST/ml_dataset → MODEL/)")
    run([sys.executable, os.path.join(BASE, "ml", "train_model.py"),
         "--dataset-dir", DEST_ML, "--model-dir", MODELS])
    print("[OK] Modèle réentraîné — le Consumer le chargera à son prochain démarrage.")


def run_producer_continuous(batch_size: int, interval: int) -> subprocess.Popen:
    """Flux CONTINU : veille de SOURCE/continuous_ingress, déplacement au fil de l'eau."""
    log("CONTINU — Producer continuous (arrière-plan)")
    check_jar(PRODUCER_JAR)
    cmd = SPARK_SUBMIT_ARGS + [
        "--class", "com.project.producer.ProducerApp", PRODUCER_JAR,
        SOURCE_INGRESS, DEST_INGRESS, str(batch_size), str(max(1, interval)), "true", "continuous",
    ]
    print(f"[CMD] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    print(f"[INFO] Producer continuous PID : {proc.pid}")
    return proc


def run_consumer() -> subprocess.Popen:
    log("CONTINU — Consumer Structured Streaming (arrière-plan)")
    check_jar(CONSUMER_JAR)
    cmd = SPARK_SUBMIT_ARGS + [
        "--class", "com.project.consumer.ConsumerApp", CONSUMER_JAR,
        DEST_INGRESS, STAGING, OUTPUT, MODELS, CHECKPOINT,
    ]
    print(f"[CMD] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    print(f"[INFO] Consumer PID : {proc.pid}")
    return proc


def run_streamlit() -> subprocess.Popen:
    log("CONTINU — Streamlit dashboard → http://localhost:8501")
    app = os.path.join(BASE, "streamlit", "app.py")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", app,
                             "--server.port", "8501", "--server.headless", "false"])
    print(f"[INFO] Streamlit PID : {proc.pid}")
    return proc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plate Detection — orchestrateur pipeline")
    parser.add_argument("--weekly", action="store_true",
                        help="Flux hebdomadaire : Producer batch + réentraînement ML, puis exit")
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="Dépose N images test dans SOURCE/continuous_ingress (simulation)")
    parser.add_argument("--skip-build", action="store_true", help="Ne pas rebuilder les jars")
    parser.add_argument("--streamlit-only", action="store_true", help="Dashboard seulement")
    parser.add_argument("--batch-size", type=int, default=100, help="Fichiers par lot Producer")
    parser.add_argument("--interval", type=int, default=5, help="Secondes entre lots Producer")
    args = parser.parse_args()

    ensure_dirs()

    if args.streamlit_only:
        st_proc = run_streamlit()
        try:
            st_proc.wait()
        except KeyboardInterrupt:
            st_proc.terminate()
        return

    if not args.skip_build:
        build()
    else:
        print("[SKIP] Build ignoré.")

    # ─── Flux hebdomadaire : batch + réentraînement, puis exit ───────────────
    if args.weekly:
        run_producer_batch(args.batch_size, 0)
        run_training()
        return

    # ─── Flux continu : producer + consumer + dashboard ──────────────────────
    if args.seed > 0:
        seed_ingress(args.seed)

    producer_proc = run_producer_continuous(args.batch_size, args.interval)
    consumer_proc = run_consumer()

    print("[INFO] Attente des premières données dans data/output/ ...")
    waited = 0
    while waited < 60:
        if any(f.endswith(".parquet") for _, _, files in os.walk(OUTPUT) for f in files):
            break
        time.sleep(3)
        waited += 3
        print(f"  ... {waited}s")

    st_proc = run_streamlit()
    print("\n[INFO] Dashboard : http://localhost:8501")
    print("[INFO] Ctrl+C pour tout arrêter.\n")

    try:
        st_proc.wait()
    except KeyboardInterrupt:
        print("\n[STOP] Arrêt demandé.")
    finally:
        st_proc.terminate()
        consumer_proc.terminate()
        producer_proc.terminate()
        print("[STOP] Producer, Consumer et Streamlit arrêtés.")


if __name__ == "__main__":
    main()
