"""
main.py — orchestrates the two pipeline flows.

CONTINUOUS flow (default) — live inference:
  1. build fat-jars (skipped with --skip-build)
  2. Consumer, Structured Streaming (background, PERMANENT):
       DEST/continuous_ingress -> staging -> PREDICT (ONNX) -> OUTPUT
  3. Producer "continuous": EPHEMERAL job relaunched every --producer-every
     seconds (this script acts as the trigger) — each cycle moves
     SOURCE/continuous_ingress -> DEST/continuous_ingress then exits
  4. Streamlit dashboard -> http://localhost:8501

WEEKLY flow (--weekly) — model retraining:
  1. Producer "batch": SOURCE/ml_dataset -> DEST/ml_dataset (recursive copy)
  2. ml/train_yolo.py: YOLOv8n fine-tune + IoU eval + ONNX export + metrics

Usage:
    python main.py                     # full continuous flow
    python main.py --weekly            # weekly batch + training, then exit
    python main.py --seed 50           # drop 50 test images into the ingress
    python main.py --skip-build
    python main.py --streamlit-only

Weekly scheduling: schtasks (Windows) or cron on `python main.py --weekly`.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import threading
import time

# Windows cp1252 console: avoid UnicodeEncodeError when stdout is redirected
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

SOURCE_INGRESS = os.path.join(DATA, "source", "continuous_ingress")
SOURCE_ML      = os.path.join(DATA, "source", "ml_dataset")
DEST_INGRESS   = os.path.join(DATA, "dest", "continuous_ingress")
DEST_ML        = os.path.join(DATA, "dest", "ml_dataset")
ARCHIVE        = os.path.join(DATA, "archive", "continuous_ingress")
STAGING        = os.path.join(DATA, "staging")
OUTPUT         = os.path.join(DATA, "output")
MODELS         = os.path.join(DATA, "models")
CHECKPOINT     = os.path.join(DATA, "checkpoint")

# bundled dataset plays SOURCE/ml_dataset if that dir is absent
DATASET_FALLBACK = os.path.join(BASE, "license-plate-detection-dataset-10125-images")

PRODUCER_JAR = os.path.join(BASE, "producer", "target", "scala-2.13", "producer-assembly-0.1.0.jar")
CONSUMER_JAR = os.path.join(BASE, "consumer", "target", "scala-2.13", "consumer-assembly-0.1.0.jar")
SBT_JAR      = os.path.join(BASE, "sbt-launch.jar")

# shutil.which resolves spark-submit.cmd on Windows (Popen alone does not)
SPARK_SUBMIT = shutil.which("spark-submit") or "spark-submit"

SPARK_SUBMIT_ARGS = [
    SPARK_SUBMIT,
    "--master", "local[4]",
    "--driver-memory", "2g",
    "--conf", "spark.driver.extraJavaOptions=-Djava.awt.headless=true",
    # force loopback: virtual adapters (VPN/WSL) break driver<->executor RPC
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
    for d in [SOURCE_INGRESS, DEST_INGRESS, DEST_ML, ARCHIVE, STAGING, OUTPUT, MODELS, CHECKPOINT]:
        os.makedirs(d, exist_ok=True)


def ml_source_dir() -> str:
    if os.path.isdir(SOURCE_ML):
        return SOURCE_ML
    if os.path.isdir(DATASET_FALLBACK):
        print(f"[INFO] SOURCE/ml_dataset missing — using bundled dataset: {DATASET_FALLBACK}")
        return DATASET_FALLBACK
    print(f"[ERROR] no dataset: neither {SOURCE_ML} nor {DATASET_FALLBACK}")
    sys.exit(1)


def check_jar(jar: str):
    if not os.path.exists(jar):
        print(f"[ERROR] jar not found: {jar}\n  -> rerun without --skip-build")
        sys.exit(1)


# ─── Steps ───────────────────────────────────────────────────────────────────

def build():
    log("BUILD — fat-jars (sbt assembly)")
    if not os.path.exists(SBT_JAR):
        print(f"[ERROR] sbt-launch.jar not found: {SBT_JAR}")
        sys.exit(1)
    run(["java", "-Xmx512m", "-jar", SBT_JAR, "assembly"])
    for jar in [PRODUCER_JAR, CONSUMER_JAR]:
        check_jar(jar)
    print("[OK] jars built.")


def seed_ingress(n: int):
    """Simulate arrivals: copy n test images into SOURCE/continuous_ingress.
    Images only — the ingress never contains YOLO labels."""
    log(f"SEED — {n} test images -> {SOURCE_INGRESS}")
    src = os.path.join(ml_source_dir(), "test", "images")
    if not os.path.isdir(src):
        print(f"[ERROR] no test images in {src}")
        sys.exit(1)
    images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    picked = random.sample(images, min(n, len(images)))
    for name in picked:
        shutil.copy2(os.path.join(src, name), os.path.join(SOURCE_INGRESS, name))
    print(f"[OK] {len(picked)} images dropped into SOURCE/continuous_ingress.")


def run_producer_batch(batch_size: int, interval: int):
    """WEEKLY: full recursive copy of ml_dataset (tree preserved)."""
    log("WEEKLY 1/2 — batch Producer: SOURCE/ml_dataset -> DEST/ml_dataset")
    check_jar(PRODUCER_JAR)
    run(SPARK_SUBMIT_ARGS + [
        "--class", "com.project.producer.ProducerApp", PRODUCER_JAR,
        ml_source_dir(), DEST_ML, str(batch_size), str(interval), "true", "batch",
    ])
    print("[OK] weekly copy done.")


def run_training():
    log("WEEKLY 2/2 — YOLO training (DEST/ml_dataset -> MODEL/)")
    run([sys.executable, os.path.join(BASE, "ml", "train_yolo.py"),
         "--dataset-dir", DEST_ML, "--model-dir", MODELS])
    print("[OK] YOLO retrained — the Consumer loads it on next start.")


def run_producer_ingress(batch_size: int, interval: int):
    """One CONTINUOUS cycle: ephemeral Producer job — scan, move, exit."""
    check_jar(PRODUCER_JAR)
    run(SPARK_SUBMIT_ARGS + [
        "--class", "com.project.producer.ProducerApp", PRODUCER_JAR,
        SOURCE_INGRESS, DEST_INGRESS, str(batch_size), str(interval), "true", "continuous",
    ])


def producer_scheduler(stop_event: threading.Event, every: int, batch_size: int, interval: int):
    """External trigger role: relaunch the ephemeral Producer every `every`
    seconds. The Producer itself never loops."""
    while not stop_event.is_set():
        try:
            run_producer_ingress(batch_size, interval)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] Producer cycle failed (code {exc.returncode}) — retrying next tick.")
        stop_event.wait(every)


def run_consumer() -> subprocess.Popen:
    log("CONTINUOUS — Consumer Structured Streaming (background)")
    check_jar(CONSUMER_JAR)
    cmd = SPARK_SUBMIT_ARGS + [
        "--class", "com.project.consumer.ConsumerApp", CONSUMER_JAR,
        DEST_INGRESS, STAGING, OUTPUT, MODELS, CHECKPOINT, ARCHIVE,
    ]
    print(f"[CMD] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    print(f"[INFO] Consumer PID: {proc.pid}")
    return proc


def run_streamlit() -> subprocess.Popen:
    log("CONTINUOUS — Streamlit dashboard -> http://localhost:8501")
    app = os.path.join(BASE, "streamlit", "app.py")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", app,
                             "--server.port", "8501", "--server.headless", "false"])
    print(f"[INFO] Streamlit PID: {proc.pid}")
    return proc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plate Detection — pipeline orchestrator")
    parser.add_argument("--weekly", action="store_true",
                        help="weekly flow: batch Producer + ML training, then exit")
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="drop N test images into SOURCE/continuous_ingress")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--streamlit-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--interval", type=int, default=0,
                        help="seconds between Producer lots")
    parser.add_argument("--producer-every", type=int, default=60, metavar="S",
                        help="relaunch period of the ephemeral continuous Producer")
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
        print("[SKIP] build skipped.")

    if args.weekly:
        run_producer_batch(args.batch_size, 0)
        run_training()
        return

    # continuous flow: permanent consumer + periodically relaunched producer
    if args.seed > 0:
        seed_ingress(args.seed)

    consumer_proc = run_consumer()

    log(f"CONTINUOUS — ephemeral Producer relaunched every {args.producer_every}s")
    stop_event = threading.Event()
    scheduler = threading.Thread(
        target=producer_scheduler,
        args=(stop_event, args.producer_every, args.batch_size, args.interval),
        daemon=True)
    scheduler.start()

    print("[INFO] waiting for first data in data/output/ ...")
    waited = 0
    while waited < 60:
        if any(f.endswith(".parquet") for _, _, files in os.walk(OUTPUT) for f in files):
            break
        time.sleep(3)
        waited += 3
        print(f"  ... {waited}s")

    st_proc = run_streamlit()
    print("\n[INFO] dashboard: http://localhost:8501")
    print("[INFO] Ctrl+C stops everything.\n")

    try:
        st_proc.wait()
    except KeyboardInterrupt:
        print("\n[STOP] shutdown requested.")
    finally:
        stop_event.set()
        st_proc.terminate()
        consumer_proc.terminate()
        print("[STOP] Producer scheduler, Consumer and Streamlit stopped.")


if __name__ == "__main__":
    main()
