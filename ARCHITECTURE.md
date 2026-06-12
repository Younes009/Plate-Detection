# Architecture & Flux — License Plate Detection

## Vue d'ensemble

```
license-plate-detection-dataset-10125-images/
        (train|valid|test)/(images|labels) — format YOLO
                │
                ▼
┌─────────────────────────────────────────────┐
│ Pipeline Spark (Scala, HDFS)                 │
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
```

## Flux détaillé (HDFS)

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
├── build.sbt
├── common/src/main/scala/Utils.scala      ← Schémas, chemins HDFS, prétraitement image
├── producer/src/main/scala/ProducerApp.scala   ← Batch : images+labels YOLO -> letterbox -> raw/
├── consumer/src/main/scala/ConsumerApp.scala   ← Streaming : raw/ -> filtre/enrichit -> processed/
└── streamlit/app.py                       ← Dashboard visu uniquement <- processed/
```

## HDFS structure obligatoire (`.claude/prompt.md`)

```
hdfs:///project/raw/
hdfs:///project/processed/
hdfs:///project/processed/aggregates/
hdfs:///project/staging/dataset/
hdfs:///project/logs/producer/
hdfs:///project/checkpoint/processed/
hdfs:///project/checkpoint/aggregates/
```
