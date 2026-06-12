# 🚗 License Plate Detection — Pipeline Spark Batch + Streaming + ML

Pipeline Big Data Scala/Spark conforme à `.claude/prompt.md` :
**Batch ingestion (YOLO/COCO/Pascal VOC) → HDFS RAW → Structured Streaming + scoring ML → HDFS PROCESSED (+ predictions) → Streamlit (lecture seule)**.

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour le schéma d'archi et le flux de données détaillé.

## Structure
```
plate-detection-spark/
├── build.sbt
├── common/src/main/scala/com/project/common/Utils.scala     # paths HDFS, schema, letterbox resize, features, parsers d'annotations
├── ml/src/main/scala/com/project/ml/ModelLoader.scala        # scoring ML (MLlib PipelineModel ou simulé)
├── producer/src/main/scala/com/project/producer/ProducerApp.scala   # batch -> raw/
├── consumer/src/main/scala/com/project/consumer/ConsumerApp.scala   # streaming raw/ -> processed/ + processed/predictions/
└── streamlit/app.py                                           # dashboard lecture seule <- processed/, processed/predictions/
```

## HDFS
```
hdfs:///project/staging/dataset/        # dataset source (train|valid|test)/(images|labels)
hdfs:///project/raw/                    # sortie producer (parquet, snappy)
hdfs:///project/processed/              # sortie consumer (parquet, append) : features + prediction + confidence_score
hdfs:///project/processed/predictions/  # sous-ensemble (image_path, prediction, confidence_score)
hdfs:///project/logs/producer/          # log d'exécution producer (parquet, append)
hdfs:///project/checkpoint/processed/   # checkpoint streaming (flux principal)
hdfs:///project/checkpoint/predictions/ # checkpoint streaming (flux predictions)
hdfs:///project/models/plate_quality_model/  # (optionnel) PipelineModel MLlib pré-entraîné
```

## Build
```bash
sbt clean assembly
```
Génère `producer/target/scala-2.12/producer-assembly-0.1.0.jar` et `consumer/target/scala-2.12/consumer-assembly-0.1.0.jar`.

## Run

### Producer (batch, une seule exécution)
```bash
spark-submit --class com.project.producer.ProducerApp --master yarn \
  --deploy-mode cluster \
  --driver-memory 4g --executor-memory 4g --executor-cores 2 \
  producer/target/scala-2.12/producer-assembly-0.1.0.jar \
  hdfs:///project/staging/dataset
```
- Lit les images (`*.jpg/.jpeg/.png`) et annotations dans `<dataset>/{train,valid,test}/{images,labels}/` :
  - **YOLO** : `labels/*.txt` (`class_id cx cy w h` normalisé) — classe optionnelle via `classes.txt` à la racine.
  - **COCO** : `_annotations.coco.json` (images/annotations/categories, bbox pixels `[x,y,w,h]`).
  - **Pascal VOC** : `labels/*.xml` (`<annotation><size>...<object><name/><bndbox>...`).
- Letterbox resize 256x256 (Java AWT/ImageIO) + reprojection des bounding boxes.
- Calcule les features bbox : coordonnées absolues, aires, ratio, quadrants, `num_boxes`.
- Écrit une seule fois en Parquet (Snappy) dans `hdfs:///project/raw/` puis s'arrête (`spark.stop()`).

Sans argument, le dataset par défaut est `hdfs:///project/staging/dataset`.

### Consumer (streaming, tourne en continu)
```bash
spark-submit --class com.project.consumer.ConsumerApp --master yarn \
  --deploy-mode cluster \
  --driver-memory 2g --executor-memory 4g --executor-cores 2 \
  --num-executors 2 \
  consumer/target/scala-2.12/consumer-assembly-0.1.0.jar
```
- **Flux 1 (data)** : `readStream` (schéma explicite) sur `hdfs:///project/raw/`.
- Filtre les images sans détection (`num_boxes > 0`), ajoute `quality_flag` (`OK` / `TOO_SMALL` / `BAD_RATIO`) et `processed_at`.
- **Flux 2 (modèle ML)** : `ModelLoader` charge un `PipelineModel` MLlib depuis `hdfs:///project/models/plate_quality_model` si présent, sinon applique un scoring simulé réaliste basé sur les features bbox. Ajoute `prediction` (string) et `confidence_score` (double).
- **Double écriture** :
  1. flux enrichi complet -> `hdfs:///project/processed/` (append, checkpoint `checkpoint/processed/`)
  2. sous-ensemble `(image_path, prediction, confidence_score)` -> `hdfs:///project/processed/predictions/` (append, checkpoint `checkpoint/predictions/`)
- Checkpointing strict (exactly-once), tourne indéfiniment via `awaitAnyTermination()`.

### Dashboard (lecture seule)
```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```
- Lit `hdfs:///project/processed/` et `hdfs:///project/processed/predictions/`.
- Filtres : `split`, `label` (classe), `quality_flag`, seuil `confidence_score`.
- KPIs : nb plaques détectées, images distinctes, confidence moyenne, plaques moyennes/image.
- Charts : répartition par classe, par prédiction, par taille de bbox, par quality_flag, histogramme `confidence_score`.
- Bouton de rafraîchissement. Aucun calcul ML, aucune écriture : lecture seule.

## Schéma `raw/` (producer) / `processed/` (consumer, + `prediction`/`confidence_score`)
| Colonne | Description |
|---|---|
| `image_name`, `split`, `image_path` | Identifiants source |
| `label` | Nom de la classe (YOLO `classes.txt` / COCO `categories` / VOC `<name>`, sinon `class_<id>`) |
| `orig_width`, `orig_height` | Dimensions originales de l'image |
| `image_data` | Image letterboxée 256x256, JPEG (binaire) |
| `class_id` | Identifiant de classe (nullable si pas de bbox) |
| `cx_norm`, `cy_norm`, `w_norm`, `h_norm` | Bbox normalisée, reprojetée après letterbox |
| `x_min`, `y_min`, `x_max`, `y_max` | Bbox en pixels (image 256x256) |
| `bbox_area`, `bbox_area_norm`, `aspect_ratio` | Features bbox |
| `bbox_size_category` | small / medium / large |
| `center_x_quadrant`, `center_y_quadrant` | left/right, top/bottom |
| `num_boxes` | Nb de plaques détectées sur l'image |
| `ingested_at` | Timestamp d'écriture par le producer |
| `quality_flag` *(processed/ uniquement)* | OK / TOO_SMALL / BAD_RATIO |
| `processed_at` *(processed/ uniquement)* | Timestamp de traitement par le consumer |
| `prediction` *(processed/ uniquement)* | `plate` / `uncertain` / `background` (ModelLoader) |
| `confidence_score` *(processed/ uniquement)* | Score de confiance [0,1] (ModelLoader) |

## Flux de données
1. **Producer** lit images + annotations (YOLO/COCO/Pascal VOC) du dataset, letterbox resize 256x256, reprojette les bbox, calcule les features, écrit une fois en Parquet (Snappy) dans `raw/`, s'arrête.
2. **Consumer** lit en streaming `raw/` (micro-batch 10s), filtre + enrichit (`quality_flag`), applique le scoring ML (`ModelLoader` : MLlib ou simulé) pour ajouter `prediction`/`confidence_score`, écrit en append dans `processed/` et `processed/predictions/`, checkpoint sous `checkpoint/`. Tourne indéfiniment via `awaitAnyTermination()`.
3. **Streamlit** (`streamlit/app.py`) lit uniquement `processed/` et `processed/predictions/` (parquet) : aucun calcul ML, uniquement KPIs/charts/filtres/tables.
