# Architecture — Plate Detection Spark Pipeline

## Vue d'ensemble

```
Dataset  ──batch──▶  data/dest/  ──stream──▶  data/output/  ──read──▶  Dashboard
(disk)    Producer    (Parquet)    Consumer     (Parquet)     Streamlit   (UI)
```

---

## Schéma détaillé

```
╔══════════════════════════════════════════════════════════════════════════╗
║                   PLATE DETECTION — ARCHITECTURE                         ║
╚══════════════════════════════════════════════════════════════════════════╝

  LICENSE PLATE DATASET
  ┌─────────────────────────────────┐
  │  train/images/*.jpg  (7 057)    │
  │  train/labels/*.txt  (7 057)    │
  │  valid/images/*.jpg  (2 048)    │   YOLO format:
  │  valid/labels/*.txt  (2 048)    │   "0 cx cy w h\n..."
  │  test/images/*.jpg   (1 020)    │   1 classe : license-plate
  │  test/labels/*.txt   (1 020)    │
  └─────────────┬───────────────────┘
                │ listImages() récursif
                ▼
╔══════════════════════════════════╗
║  MICROSERVICE 1 — PRODUCER       ║  Spark Core (Batch)
║  ProducerApp.scala               ║  spark-submit --master local[2]
╠══════════════════════════════════╣
║  1. Liste toutes les images      ║
║  2. Groupe par batchSize (200)   ║
║  3. spark.read.format(binaryFile)║
║  4. Extrait image_name, split    ║  ← AUCUNE TRANSFORMATION
║  5. Écrit Parquet Snappy         ║  ← IMAGES BRUTES UNIQUEMENT
║  6. Sleep(intervalSeconds)       ║
║  7. Batch suivant → stop à fin   ║
╠══════════════════════════════════╣
║  Schema écrit (destSchema) :     ║
║  image_name | split              ║
║  image_path | image_data (bytes) ║
║  ingested_at                     ║
╚══════════════╤═══════════════════╝
               │ Parquet Snappy
               │ batch par batch (goute à goute)
               ▼
  ┌────────────────────────────────┐
  │         data/dest/             │
  │   batch_0_<ts>/  *.parquet     │
  │   batch_1_<ts>/  *.parquet     │  ← Structured Streaming source
  │   batch_N_<ts>/  *.parquet     │    (recursiveFileLookup=true)
  └────────────┬───────────────────┘
               │ readStream (maxFilesPerTrigger=1)
               │ trigger toutes les 10s
               ▼
╔══════════════════════════════════╗
║  MICROSERVICE 2 — CONSUMER       ║  Spark Structured Streaming
║  ConsumerApp.scala               ║  spark-submit --master local[2]
╠══════════════════════════════════╣
║  readStream (écoute permanente)  ║
║  ┌── foreachBatch ─────────────┐ ║
║  │ 1. Joint labels YOLO        │ ║  image_path → /labels/ → .txt
║  │    image_path → label_path  │ ║  UDF readLabel(file:///...)
║  │    → label_data (texte)     │ ║
║  │ 2. ModelLoader.score()      │ ║──────────────────────┐
║  │    prediction               │ ║                      │
║  │    confidence_score         │ ║                      │
║  │ 3. Écrit output/            │ ║                      │
║  │ 4. Écrit output/predictions/│ ║                      │
║  └─────────────────────────────┘ ║                      │
║  awaitTermination()              ║                      │
╚════════════════════════════════╤═╝                      │
                                 │                        │
              ┌──────────────────┘             ╔══════════╧══════════╗
              │                                ║  ML MODULE          ║
              ▼                                ║  ModelLoader.scala  ║
  ┌───────────────────────────┐                ╠═════════════════════╣
  │     data/output/          │                ║ 1. best_model.onnx  ║
  │     *.parquet             │                ║    PyTorch → image  ║
  │                           │                ║ 2. plate_model.onnx ║
  │  image_name | split       │                ║    sklearn → bbox   ║
  │  image_path | image_data  │                ║ 3. spark_model/     ║
  │  label_path | label_data  │                ║    MLlib Pipeline   ║
  │  prediction               │                ║ 4. simulé (fallback)║
  │  confidence_score         │                ║    conf=0.5         ║
  │  processed_at             │                ╚═════════════════════╝
  │                           │
  │  predictions/ *.parquet   │
  │  image_path | label_data  │
  │  prediction | conf_score  │
  │  processed_at             │
  └───────────┬───────────────┘
              │ pandas + pyarrow (glob *.parquet)
              │ cache TTL 60s
              ▼
╔══════════════════════════════════╗
║  MICROSERVICE 3 — DASHBOARD      ║  Streamlit
║  streamlit/app.py                ║  localhost:8501
╠══════════════════════════════════╣
║  Filtres : split / prediction    ║
║            confidence / label    ║
║                                  ║
║  KPIs : images filtrées          ║
║         images distinctes        ║
║         confidence moyenne       ║
║         images avec label YOLO   ║
║                                  ║
║  Charts : prédiction bar chart   ║
║           split bar chart        ║
║           distribution confiance ║
║           stats bbox YOLO        ║
║                                  ║
║  Table : données enrichies       ║
║  Detail : labels YOLO bruts      ║
╚══════════════════════════════════╝
```

---

## Schemas des données

### destSchema — écrit par le Producer

| Colonne | Type | Description |
|---|---|---|
| `image_name` | String | Nom du fichier sans extension |
| `split` | String | `train` / `valid` / `test` / `unknown` |
| `image_path` | String | URI complète du fichier image |
| `image_data` | Binary | Contenu brut JPEG/PNG |
| `ingested_at` | Timestamp | Horodatage d'ingestion |

### outputSchema — écrit par le Consumer

Tout le `destSchema` plus :

| Colonne | Type | Description |
|---|---|---|
| `label_path` | String | URI du fichier `.txt` YOLO correspondant |
| `label_data` | String | Contenu YOLO brut : `"0 cx cy w h\n..."` |
| `prediction` | String | `plate` / `background` / `uncertain` |
| `confidence_score` | Double | Score de confiance `[0.0, 1.0]` |
| `processed_at` | Timestamp | Horodatage de traitement |

---

## Stack technique

| Composant | Technologie | Version |
|---|---|---|
| Producer / Consumer | Scala + Apache Spark | 2.13.14 / 4.1.1 |
| Build | sbt + sbt-assembly | 1.9.9 |
| Runtime JVM | Java | 17 |
| ML scoring | ONNX Runtime / MLlib | — |
| Dashboard | Streamlit | 1.58 |
| Lecture Parquet | pandas + pyarrow | 3.x |

---

## Lancement

```bash
# Pipeline complet
python main.py

# Sans rebuild
python main.py --skip-build

# Producer seul (depuis la racine du projet)
spark-submit --master local[2] \
  producer/target/scala-2.13/producer-assembly-0.1.0.jar \
  license-plate-detection-dataset-10125-images \
  data/dest 200 5

# Consumer seul
spark-submit --master local[2] \
  consumer/target/scala-2.13/consumer-assembly-0.1.0.jar \
  data/dest data/output data/models data/checkpoint

# Dashboard seul
python main.py --streamlit-only
```
