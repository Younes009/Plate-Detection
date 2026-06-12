# Architecture & Flux — License Plate Detection

## Vue d'ensemble

```
hdfs:///project/staging/dataset/
        (train|valid|test)/(images|labels) — YOLO .txt, COCO .json, ou Pascal VOC .xml
                │
                ▼
┌──────────────────────────────────────────────────────────┐
│ Pipeline Spark (Scala, packages com.project.*, HDFS)      │
│                                                            │
│  ProducerApp (batch, 1x)            com.project.producer  │
│   - lit images + annotations (YOLO/COCO/Pascal VOC)       │
│   - letterbox resize 256x256, reprojette bbox             │
│   - calcule features (aire, ratio, quadrants, num_boxes)  │
│   - write parquet (snappy) -> hdfs:///project/raw/        │
│   - log run -> hdfs:///project/logs/producer/             │
│   - spark.stop()                                          │
│                                                            │
│  ModelLoader                        com.project.ml        │
│   - PipelineModel MLlib (hdfs:///project/models/...)      │
│     si dispo, sinon scoring simulé réaliste               │
│   - ajoute prediction + confidence_score                  │
│                                                            │
│  ConsumerApp (streaming, infini)    com.project.consumer  │
│   - readStream raw/ (parquet, schéma explicite)           │
│   - filter num_boxes>0, ajoute quality_flag, processed_at │
│   - ModelLoader.score() -> prediction, confidence_score   │
│   - double écriture (append) :                            │
│       processed/             (flux enrichi complet)       │
│       processed/predictions/ (image_path,prediction,...)  │
│   - checkpoint/processed/, checkpoint/predictions/        │
│   - awaitAnyTermination() (exactly-once)                  │
│                                                            │
│  streamlit/app.py (lecture seule)                         │
│   - lit processed/ + processed/predictions/               │
│   - KPIs, charts, histogramme confidence, filtres, refresh│
└──────────────────────────────────────────────────────────┘
```

## Flux détaillé (HDFS)

```
hdfs:///project/staging/dataset/{train,valid,test}/{images,labels}
   images : *.jpg|*.jpeg|*.png
   labels : YOLO *.txt ("class_id cx cy w h") | Pascal VOC *.xml | COCO _annotations.coco.json
        │
        ▼ ProducerApp (batch)
hdfs:///project/raw/                <- parquet (snappy), schema preprocessedSchema
   image_name, split, image_path, label,
   orig_width/height, image_data(256x256 jpg),
   class_id, cx/cy/w/h_norm, x_min..y_max,
   bbox_area(_norm), aspect_ratio,
   bbox_size_category, center_x/y_quadrant,
   num_boxes, ingested_at

hdfs:///project/logs/producer/      <- parquet append, 1 ligne/run
   run_at, dataset_path, nb_images, nb_rows
        │
        ▼ ConsumerApp (streaming, micro-batch 10s)
   filter num_boxes>0
   + quality_flag (OK/TOO_SMALL/BAD_RATIO), processed_at
   + ModelLoader.score() -> prediction, confidence_score
        │
        ├──▶ hdfs:///project/processed/              (append, checkpoint/processed/)
        │       toutes les colonnes ci-dessus + quality_flag, processed_at,
        │       prediction, confidence_score
        │
        └──▶ hdfs:///project/processed/predictions/  (append, checkpoint/predictions/)
                image_path, prediction, confidence_score
        │
        ▼ streamlit/app.py (lecture seule)
   KPIs, charts (label/prediction/bbox_size/quality_flag),
   histogramme confidence_score, filtres (split/label/quality/confidence), refresh
```

## Structure des dossiers

```
plate-detection-spark/
├── build.sbt                                                   ← modules common, ml, producer, consumer (spark-mllib, scala-xml)
├── common/src/main/scala/com/project/common/Utils.scala        ← schémas, chemins HDFS, prétraitement image, parsers YOLO/COCO/VOC
├── ml/src/main/scala/com/project/ml/ModelLoader.scala          ← scoring ML (PipelineModel MLlib ou simulé)
├── producer/src/main/scala/com/project/producer/ProducerApp.scala  ← Batch : images+annotations -> letterbox -> raw/
├── consumer/src/main/scala/com/project/consumer/ConsumerApp.scala  ← Streaming : raw/ -> ML scoring -> processed/ + processed/predictions/
└── streamlit/app.py                                             ← Dashboard lecture seule <- processed/, processed/predictions/
```

## HDFS structure (`.claude/prompt.md`)

```
hdfs:///project/raw/
hdfs:///project/processed/
hdfs:///project/processed/predictions/
hdfs:///project/staging/dataset/
hdfs:///project/logs/producer/
hdfs:///project/checkpoint/processed/
hdfs:///project/checkpoint/predictions/
hdfs:///project/models/plate_quality_model/   (optionnel, PipelineModel MLlib)
```
