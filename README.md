# 🚗 License Plate Detection — Pipeline Spark Batch + Streaming

Pipeline Big Data Scala/Spark conforme à `.claude/prompt.md` :
**Batch ingestion → HDFS RAW → Streaming processing → HDFS PROCESSED → Streamlit (visualisation)**.

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour le schéma d'archi et le flux de données détaillé.

## Structure
```
plate-detection-spark/
├── build.sbt
├── common/src/main/scala/Utils.scala      # paths HDFS, schema, letterbox resize, features
├── producer/src/main/scala/ProducerApp.scala   # batch -> raw/
├── consumer/src/main/scala/ConsumerApp.scala   # streaming raw/ -> processed/
└── streamlit/app.py                       # dashboard visu uniquement <- processed/
```

## HDFS
```
hdfs:///project/staging/dataset/   # dataset source (train|valid|test)/(images|labels)
hdfs:///project/raw/               # sortie producer (parquet)
hdfs:///project/processed/         # sortie consumer (parquet, append)
hdfs:///project/logs/producer/     # log d'exécution producer (parquet, append)
hdfs:///project/checkpoint/        # checkpoints streaming
```

## Build
```bash
sbt clean assembly
```
Génère `producer/target/scala-2.12/producer-assembly-0.1.0.jar` et `consumer/target/scala-2.12/consumer-assembly-0.1.0.jar`.

## Run

### Producer (batch, une seule exécution)
```bash
spark-submit --class ProducerApp --master yarn \
  producer/target/scala-2.12/producer-assembly-0.1.0.jar \
  hdfs:///project/staging/dataset
```
- Lit les images (`*.jpg/.jpeg/.png`) et labels YOLO (`*.txt`) dans `<dataset>/{train,valid,test}/{images,labels}/`.
- Letterbox resize 256x256 (Java AWT/ImageIO) + reprojection des bounding boxes.
- Calcule les features bbox : coordonnées absolues, aires, ratio, quadrants, `num_boxes`.
- Écrit une seule fois en Parquet dans `hdfs:///project/raw/` puis s'arrête (`spark.stop()`).

Sans argument, le dataset par défaut est `hdfs:///project/staging/dataset`.

### Consumer (streaming, tourne en continu)
```bash
spark-submit --class ConsumerApp --master yarn \
  consumer/target/scala-2.12/consumer-assembly-0.1.0.jar
```
- `readStream` (schema explicite) sur `hdfs:///project/raw/`.
- Filtre les images sans détection (`num_boxes > 0`), ajoute `quality_flag` (`OK` / `TOO_SMALL` / `BAD_RATIO`) et `processed_at`.
- Écrit en append dans `hdfs:///project/processed/` avec checkpoint `hdfs:///project/checkpoint/processed/`.
- 2ème requête streaming : agrégats par fenêtre 1min / `split` / `bbox_size_category` (nb images, ratio moyen, nb boxes moyen) vers `hdfs:///project/processed/aggregates/`.
- Tourne indéfiniment via `awaitAnyTermination()`.

### Dashboard (visualisation uniquement)
```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```
- Lit `hdfs:///project/processed/` (table par bbox) et `hdfs:///project/processed/aggregates/` (agrégats fenêtrés).
- Filtres : `split`, `bbox_size_category`, `quality_flag`.
- Charts : répartition par taille de bbox, quality_flag, quadrants, distribution `aspect_ratio`, série temporelle des agrégats.
- Aucun calcul de scoring/ML, aucune écriture : lecture seule de `processed/`.

## Schéma `raw/` / `processed/` (une ligne = une bbox, ou une image sans bbox)
| Colonne | Description |
|---|---|
| `image_name`, `split`, `image_path` | Identifiants source |
| `orig_width`, `orig_height` | Dimensions originales de l'image |
| `image_data` | Image letterboxée 256x256, JPEG (binaire) |
| `class_id` | Classe YOLO (nullable si pas de bbox) |
| `cx_norm`, `cy_norm`, `w_norm`, `h_norm` | Bbox normalisée, reprojetée après letterbox |
| `x_min`, `y_min`, `x_max`, `y_max` | Bbox en pixels (image 256x256) |
| `bbox_area`, `bbox_area_norm`, `aspect_ratio` | Features bbox |
| `bbox_size_category` | small / medium / large |
| `center_x_quadrant`, `center_y_quadrant` | left/right, top/bottom |
| `num_boxes` | Nb de plaques détectées sur l'image |
| `ingested_at` | Timestamp d'écriture par le producer |

## Flux de données
1. **Producer** lit images + labels YOLO du dataset, letterbox resize 256x256, reprojette les bbox, calcule les features, écrit une fois en Parquet dans `raw/`, s'arrête.
2. **Consumer** détecte les nouveaux fichiers Parquet dans `raw/` (micro-batch toutes les 10s), filtre + enrichit (`quality_flag`), écrit en append dans `processed/`, et maintient en parallèle des agrégats fenêtrés dans `processed/aggregates/`. Tourne indéfiniment via `awaitAnyTermination()`.
3. **Streamlit** (`streamlit/app.py`) lit uniquement `processed/` et `processed/aggregates/` (parquet) : aucun calcul métier, uniquement charts/filtres/tables.
