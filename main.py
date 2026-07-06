"""
main.py — Lance le pipeline complet :
  1. Build des fat-jars (skippé si déjà présents)
  2. Producer  : ingestion batch des images → data/dest/
  3. Consumer  : Structured Streaming → data/output/
  4. Streamlit : dashboard → http://localhost:8501

Usage :
    python main.py                        # pipeline complet
    python main.py --skip-build           # skip sbt assembly
    python main.py --skip-producer        # consumer + streamlit seulement
    python main.py --streamlit-only       # streamlit seulement
"""

import argparse
import os
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

BASE        = os.path.dirname(os.path.abspath(__file__))
DATA        = os.path.join(BASE, "data")
DEST        = os.path.join(DATA, "dest")
STAGING     = os.path.join(DATA, "staging")
OUTPUT      = os.path.join(DATA, "output")
MODELS      = os.path.join(DATA, "models")
CHECKPOINT  = os.path.join(DATA, "checkpoint")
DATASET     = os.path.join(BASE, "license-plate-detection-dataset-10125-images")

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
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def run(cmd, **kwargs):
    print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def ensure_dirs():
    for d in [DEST, STAGING, OUTPUT, MODELS, CHECKPOINT]:
        os.makedirs(d, exist_ok=True)


# ─── Étapes ──────────────────────────────────────────────────────────────────

def build():
    log("STEP 1 — Build fat-jars (sbt assembly)")
    if not os.path.exists(SBT_JAR):
        print(f"[ERROR] sbt-launch.jar introuvable : {SBT_JAR}")
        sys.exit(1)
    run(["java", "-Xmx512m", "-jar", SBT_JAR, "clean", "assembly"])
    for jar in [PRODUCER_JAR, CONSUMER_JAR]:
        if not os.path.exists(jar):
            print(f"[ERROR] Jar manquant après build : {jar}")
            sys.exit(1)
    print("[OK] Jars construits.")


def run_producer(batch_size: int = 100, interval: int = 0):
    log("STEP 2 — Producer (batch ingestion)")
    if not os.path.exists(PRODUCER_JAR):
        print(f"[ERROR] Producer jar introuvable : {PRODUCER_JAR}\n  → Relance sans --skip-build")
        sys.exit(1)
    if not os.path.exists(DATASET):
        print(f"[WARN] Dataset introuvable : {DATASET}\n  → Le Producer créera le dossier source vide.")

    cmd = SPARK_SUBMIT_ARGS + [
        "--class", "com.project.producer.ProducerApp",
        PRODUCER_JAR,
        DATASET,
        DEST,
        str(batch_size),
        str(interval),
        "true",  # récursivité
    ]
    run(cmd)
    print("[OK] Producer terminé.")


def run_consumer():
    log("STEP 3 — Consumer (Structured Streaming) — Ctrl+C pour stopper")
    if not os.path.exists(CONSUMER_JAR):
        print(f"[ERROR] Consumer jar introuvable : {CONSUMER_JAR}\n  → Relance sans --skip-build")
        sys.exit(1)

    cmd = SPARK_SUBMIT_ARGS + [
        "--class", "com.project.consumer.ConsumerApp",
        CONSUMER_JAR,
        DEST,
        STAGING,
        OUTPUT,
        MODELS,
        CHECKPOINT,
    ]
    # Consumer tourne indéfiniment — on le lance en arrière-plan
    print("[INFO] Consumer lancé en arrière-plan (PID affiché ci-dessous).")
    proc = subprocess.Popen(cmd)
    print(f"[INFO] Consumer PID : {proc.pid}")
    return proc


def run_streamlit():
    log("STEP 4 — Streamlit dashboard → http://localhost:8501")
    app = os.path.join(BASE, "streamlit", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app,
           "--server.port", "8501", "--server.headless", "false"]
    proc = subprocess.Popen(cmd)
    print(f"[INFO] Streamlit PID : {proc.pid}")
    return proc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plate Detection — pipeline complet")
    parser.add_argument("--skip-build",    action="store_true", help="Ne pas rebuilder les jars")
    parser.add_argument("--skip-producer", action="store_true", help="Ne pas relancer le Producer")
    parser.add_argument("--streamlit-only",action="store_true", help="Lancer uniquement Streamlit")
    parser.add_argument("--batch-size",    type=int, default=100, help="Images par batch Producer (défaut 100)")
    parser.add_argument("--interval",      type=int, default=0,   help="Secondes entre batches (défaut 0)")
    args = parser.parse_args()

    ensure_dirs()

    if args.streamlit_only:
        st_proc = run_streamlit()
        try:
            st_proc.wait()
        except KeyboardInterrupt:
            st_proc.terminate()
        return

    # Build
    if not args.skip_build:
        build()
    else:
        print("[SKIP] Build ignoré.")

    # Producer
    if not args.skip_producer:
        run_producer(args.batch_size, args.interval)
    else:
        print("[SKIP] Producer ignoré.")

    # Consumer (background)
    consumer_proc = run_consumer()

    # Attendre que des données apparaissent avant de lancer Streamlit
    print("[INFO] Attente données dans data/output/ ...")
    waited = 0
    while waited < 60:
        if any(f.endswith(".parquet") for _, _, files in os.walk(OUTPUT) for f in files):
            break
        time.sleep(3)
        waited += 3
        print(f"  ... {waited}s")

    # Streamlit (foreground — bloque jusqu'à Ctrl+C)
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
        print("[STOP] Consumer et Streamlit arrêtés.")


if __name__ == "__main__":
    main()
