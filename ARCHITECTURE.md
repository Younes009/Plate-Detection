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
└─────────────────────────────────────────────┘
                │ processed/ (parquet: image_data, bbox feats, quality_flag)
                ▼
┌─────────────────────────────────────────────┐
│ 2-model-training (Python)                    │
│  dataloader.py -> model.py (YOLO-like)       │
│  train.py -> checkpoints/, best.pt           │
└─────────────────────────────────────────────┘
                │ best.pt
                ▼
┌─────────────────────────────────────────────┐
│ 3-web-interface (Flask)                      │
│  app.py / api.py -> inference.py (best.pt)   │
│  spark_batch.py -> stats/EDA                 │
│  history.csv (log prédictions + images/)     │
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
        │
        ▼ ConsumerApp (streaming, micro-batch 10s)
   ├─ filter num_boxes>0 + quality_flag + processed_at
   │       ▼
   │  hdfs:///project/processed/        (append, checkpoint/processed/)
   │
   └─ window(1min) groupBy split, bbox_size_category
           ▼
      hdfs:///project/processed/aggregates/  (append, checkpoint/aggregates/)
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
│   └── build.sbt
│
├── 2-model-training/             ← Microservice 2 : Entraînement YOLOv8
│   ├── train.py, dataloader.py, model.py, loss.py
│   └── checkpoints/, best.pt
│
└── 3-web-interface/               ← Microservice 3 : Interface Web Flask
    ├── app.py, api.py, inference.py, spark_batch.py
    ├── model/best.pt
    └── history.csv
```

## Connu / à recâbler

- `2-model-training/train.py:34` et `dataloader.py:266` pointent vers `../1-spark-preprocessing/data/processed`
  (chemin local hérité de l'ancien pipeline PySpark). La sortie réelle de `1-spark-preprocessing` est
  `hdfs:///project/processed/` (parquet) — adapter le dataloader pour lire depuis HDFS si besoin.
