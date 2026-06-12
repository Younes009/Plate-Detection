# Architecture & Flux — License Plate Detection

## Vue d'ensemble (3 microservices)

```
license-plate-detection-dataset-10125-images/
        (train|valid|test)/(images|labels) — format YOLO
                │
                ▼
┌─────────────────────────────────────────────┐
│ 1-spark-preprocessing (Scala/Spark, HDFS)    │
│                                               │
│  ProducerApp (batch, 1x)                     │
│   - lit images+labels YOLO                   │
│   - letterbox resize 256x256                 │
│   - reprojette bbox, calcule features        │
│   - write parquet -> hdfs:///project/raw/    │
│   - spark.stop()                             │
│                                               │
│  ConsumerApp (streaming, infini)             │
│   - readStream raw/ (parquet)                │
│   - filter num_boxes>0, quality_flag         │
│   - writeStream append -> processed/         │
│     + checkpoint/processed/                  │
│   - aggStream (window 1min/split/size_cat)   │
│     -> processed/aggregates/                 │
│     + checkpoint/aggregates/                 │
│                                               │
│  streamlit/app.py (visu uniquement)          │
│   - lit processed/ + processed/aggregates/   │
│   - charts / filtres / tables, pas de calcul │
└─────────────────────────────────────────────┘
                │ processed/ (parquet: image_data, bbox feats, quality_flag)
                ▼
┌─────────────────────────────────────────────┐
│ 2-model-training (Python / PyTorch)          │
│  dataloader.py -> model.py + loss.py         │
│  train.py -> checkpoints/best_model.pt       │
└─────────────────────────────────────────────┘
                │ checkpoints/best_model.pt
                ▼
┌─────────────────────────────────────────────┐
│ 3-web-interface (FastAPI + Streamlit)        │
│  api.py (FastAPI :8000) -> inference.py      │
│    -> charge ../2-model-training/            │
│       checkpoints/best_model.pt              │
│  api.py -> spark_batch.py (PySpark,          │
│             inférence batch si >3 images)    │
│  app.py (Streamlit :8501) -> appelle api.py  │
│  history.csv + images/ (log prédictions)     │
└─────────────────────────────────────────────┘
```

## Flux détaillé — `1-spark-preprocessing` (HDFS)

```
hdfs:///project/staging/dataset/{train,valid,test}/{images,labels}
        │
        ▼ ProducerApp (batch)
hdfs:///project/raw/                <- parquet, schema preprocessedSchema
   image_name, split, image_path,
   orig_width/height, image_data(256x256 jpg),
   class_id, cx/cy/w/h_norm, x_min..y_max,
   bbox_area(_norm), aspect_ratio,
   bbox_size_category, center_x/y_quadrant,
   num_boxes, ingested_at

hdfs:///project/logs/producer/      <- parquet append, 1 ligne/run
   run_at, dataset_path, nb_images, nb_rows
        │
        ▼ ConsumerApp (streaming, micro-batch 10s)
   ├─ filter num_boxes>0 + quality_flag + processed_at
   │       ▼
   │  hdfs:///project/processed/        (append, checkpoint/processed/)
   │
   └─ window(1min) groupBy split, bbox_size_category
           ▼
      hdfs:///project/processed/aggregates/  (append, checkpoint/aggregates/)
        │
        ▼ streamlit/app.py (lecture seule, charts/filtres/tables)
```

## Structure des dossiers

```
plate-detection-spark/
├── license-plate-detection-dataset-10125-images/   ← Dataset YOLO (train/valid/test)
│
├── 1-spark-preprocessing/        ← Microservice 1 : Prétraitement Spark (Scala)
│   ├── producer/   ← Batch : images+labels YOLO -> letterbox -> raw/
│   ├── consumer/   ← Streaming : raw/ -> filtre/enrichit -> processed/
│   ├── common/     ← Schémas, chemins HDFS, prétraitement image
│   ├── streamlit/app.py ← Dashboard visu uniquement <- processed/
│   └── build.sbt
│
├── 2-model-training/              ← Microservice 2 : Entraînement (PyTorch)
│   ├── train.py, dataloader.py, model.py, loss.py
│   └── checkpoints/best_model.pt
│
└── 3-web-interface/                ← Microservice 3 : FastAPI + Streamlit
    ├── api.py        ← FastAPI (port 8000) -> inference.py -> spark_batch.py
    ├── app.py        ← Streamlit (port 8501), appelle api.py
    ├── inference.py  ← charge ../2-model-training/checkpoints/best_model.pt
    ├── spark_batch.py ← inférence batch PySpark (>3 images)
    ├── run.sh        ← lance api|streamlit|both
    ├── history.csv, images/, uploads/
```

## Connu / à recâbler

- `2-model-training/train.py:34` et `dataloader.py:266` pointent vers `../1-spark-preprocessing/data/processed`
  (chemin local hérité de l'ancien pipeline PySpark). La sortie réelle de `1-spark-preprocessing` est
  `hdfs:///project/processed/` (parquet) — adapter le dataloader pour lire depuis HDFS si besoin.
