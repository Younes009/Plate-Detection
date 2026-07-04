# License Plate Detection — Pipeline Spark Batch + Streaming + ML

Pipeline Big Data Scala/Spark en deux microservices :

**Microservice 1 (Producer)** : ingestion batch récursive (YOLO/COCO/Pascal VOC) → `DEST/`  
**Microservice 2 (Consumer)** : Structured Streaming → parse/transform/score/predict → `OUTPUT/`  
**ML** : entraînement sklearn sur `OUTPUT/`, export ONNX/pickle → Consumer recharge le modèle  
**Streamlit** : dashboard lecture seule sur `OUTPUT/`

---

## Structure du projet

```
plate-detection-spark/
├── main.py                                                           # point d'entrée unique — orchestre tout le pipeline
├── build.sbt
├── common/src/main/scala/com/project/common/Utils.scala             # chemins HDFS, schémas, letterbox, features, parsers
├── producer/src/main/scala/com/project/producer/ProducerApp.scala   # Microservice 1 — batch récursif batchSize+interval
├── consumer/src/main/scala/com/project/consumer/ConsumerApp.scala   # Microservice 2 — streaming DEST/ → OUTPUT/
├── ml/
│   ├── src/main/scala/com/project/ml/ModelLoader.scala              # scoring PyTorch ONNX → sklearn ONNX → MLlib → simulé
│   ├── checkpoints/best_model.pt                                     # modèle PyTorch pré-entraîné (détection plaques)
│   ├── export_pytorch.py                                             # export best_model.pt → best_model.onnx
│   └── train_model.py                                                # entraînement sklearn sur OUTPUT/ → plate_model.onnx
└── streamlit/app.py                                                  # dashboard lecture seule ← OUTPUT/
```

---

## Dossiers HDFS

```
hdfs:///project/source/               # images + annotations source (créé automatiquement si absent)
hdfs:///project/dest/                 # sortie Producer — batch_N_TS/ (Parquet Snappy)
hdfs:///project/output/               # sortie Consumer — données enrichies + prédictions (Parquet Snappy, append)
hdfs:///project/output/predictions/   # sous-ensemble (image_path, prediction, confidence_score, processed_at)
hdfs:///project/models/               # plate_model.onnx (export Python) ou spark_model/ (PipelineModel MLlib)
hdfs:///project/checkpoint/consumer/  # checkpoint Structured Streaming (exactly-once)
hdfs:///project/logs/producer/        # log d'exécution Producer (Parquet, append)
```

---

## Lancement rapide — `main.py`

`main.py` orchestre tout le pipeline en une seule commande.

```bash
# Pipeline complet : build → producer → consumer → streamlit
python main.py

# Jars déjà buildés (gain de temps)
python main.py --skip-build

# Consumer + Streamlit seulement (dest/ déjà rempli)
python main.py --skip-build --skip-producer

# Dashboard seul (output/ déjà rempli)
python main.py --streamlit-only
```

**Options disponibles :**

| Option | Défaut | Description |
|--------|--------|-------------|
| `--skip-build` | false | Skip `sbt assembly` |
| `--skip-producer` | false | Skip le Producer |
| `--streamlit-only` | false | Lance uniquement Streamlit |
| `--batch-size N` | 100 | Images par batch Producer |
| `--interval N` | 0 | Secondes entre batches |

**Séquence exécutée par `main.py` :**
1. Build fat-jars via `sbt-launch.jar`
2. Producer en foreground → attend la fin
3. Consumer en arrière-plan → attend les premiers fichiers dans `data/output/`
4. Streamlit en foreground sur `:8501` → `Ctrl+C` arrête tout

---

## Build manuel

```bash
java -Xmx512m -jar sbt-launch.jar clean assembly
```

Produit :
- `producer/target/scala-2.13/producer-assembly-0.1.0.jar`
- `consumer/target/scala-2.13/consumer-assembly-0.1.0.jar`

> `onnxruntime 1.18.0` est bundlé dans le fat-jar du Consumer (non-provided).

---

## Exécution manuelle — ordre obligatoire

### Étape 1 — Producer (Microservice 1)

```bash
spark-submit \
  --class com.project.producer.ProducerApp \
  --master yarn --deploy-mode cluster \
  --driver-memory 4g --executor-memory 4g --executor-cores 2 \
  producer/target/scala-2.13/producer-assembly-0.1.0.jar \
  hdfs:///project/source/ \   # sourcePath  : dossier images + annotations
  hdfs:///project/dest/   \   # destPath    : où écrire les batches
  100                     \   # batchSize   : images par batch
  30                          # interval    : secondes entre batches
```

**Logique :**
- Crée `sourcePath` s'il n'existe pas.
- Parcourt récursivement `sourcePath` pour trouver toutes les images (`.jpg/.jpeg/.png`) et annotations (`.txt` YOLO / `.json` COCO / `.xml` VOC) dans n'importe quelle sous-arborescence.
- Charge les annotations **une seule fois** (cache Spark), groupe les images en batches de `batchSize`.
- Pour chaque batch :
  1. Lit les images (Spark `binaryFile`).
  2. Joint avec les annotations (left join → image sans annotation = background).
  3. Letterbox resize 256×256 (Java AWT/ImageIO) + reprojection des bbox.
  4. Calcule les features bbox : coordonnées absolues, aires, `aspect_ratio`, quadrants, `num_boxes`.
  5. Écrit en Parquet Snappy dans `destPath/batch_N_TS/`.
  6. Attend `interval` secondes avant le batch suivant.
- S'arrête automatiquement après le dernier batch (**batch pur, aucune boucle infinie**).

Formats d'annotations supportés :
| Format | Fichier attendu | Détection |
|--------|----------------|-----------|
| **YOLO** | `*.txt` (5 colonnes : `class_id cx cy w h`) | toute profondeur sous `sourcePath` |
| **COCO** | `*.json` (champs `images`, `annotations`, `categories`) | toute profondeur |
| **Pascal VOC** | `*.xml` (`<annotation><size><object><bndbox>`) | toute profondeur |

---

### Étape 2 — Consumer phase 1 (sans modèle ML)

```bash
spark-submit \
  --class com.project.consumer.ConsumerApp \
  --master yarn --deploy-mode cluster \
  --driver-memory 2g --executor-memory 4g --executor-cores 2 --num-executors 2 \
  consumer/target/scala-2.13/consumer-assembly-0.1.0.jar \
  hdfs:///project/dest/        \   # destPath
  hdfs:///project/output/      \   # outputPath
  hdfs:///project/models/      \   # modelPath (vide → scoring simulé)
  hdfs:///project/checkpoint/       # checkpointPath
```

**Logique :**
- Lit `destPath/` en **Structured Streaming** (schéma explicite `destSchema`, `maxFilesPerTrigger=1`).
- Pour chaque micro-batch (`foreachBatch`) :
  1. **Filtre** anomalies (`num_boxes` null).
  2. **Enrichit** : `quality_flag` (`OK` / `TOO_SMALL` / `BAD_RATIO` / `NO_PLATE`) + `processed_at`.
  3. **Score** : `ModelLoader` (ONNX → MLlib → simulé) → colonnes `prediction` + `confidence_score`.
  4. **Écriture 1** : flux complet → `outputPath/` (append, Parquet Snappy).
  5. **Écriture 2** : `(image_path, prediction, confidence_score, processed_at)` → `outputPath/predictions/` (append).
- Checkpoint `hdfs:///project/checkpoint/consumer/` (sémantique **exactly-once**).
- Tourne indéfiniment via `awaitTermination()`.

---

### Étape 3 — Entraînement ML

```bash
# Installer les dépendances Python
pip install pyspark scikit-learn skl2onnx onnxruntime numpy pandas

# Entraîner et exporter
python ml/train_model.py \
  --output-dir hdfs:///project/output/ \   # features produites par le Consumer
  --model-dir  ./models                \   # dossier local de sortie
  --format     both                        # onnx + pickle

# Copier le modèle ONNX vers HDFS
hdfs dfs -put models/plate_model.onnx hdfs:///project/models/
```

**Logique :**
- Lit `outputPath/` (Parquet) avec PySpark → convertit en Pandas.
- Cible : `1` si `class_id` non-null (plaque annotée), `0` sinon (background).
- Utilise la colonne `split` (`train` / `valid` / `test` / `unknown`) pour les splits — fallback aléatoire 70/15/15 si vide.
- Features : `bbox_area_norm`, `aspect_ratio`, `cx_norm`, `cy_norm`, `w_norm`, `h_norm`.
- Modèle : `RandomForestClassifier` (100 arbres, `class_weight=balanced`).
- Export **ONNX** (`skl2onnx`, `zipmap=False`) → `plate_model.onnx`.
- Export **pickle** → `plate_model.pkl` (backup Python).

---

### Étape 4 — Consumer phase 2 (avec modèle ONNX)

Relancer la **même commande** spark-submit qu'à l'étape 2.  
`ModelLoader.initialize()` détecte `plate_model.onnx` dans `modelPath/` → charge et broadcast les bytes → scoring ONNX réel via `mapPartitions` (une session ONNX Runtime par partition).

---

### Étape 5 — Dashboard Streamlit

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

- Lit `hdfs:///project/output/` et `hdfs:///project/output/predictions/` (lecture seule).
- Filtres sidebar : split, label, quality_flag, prédiction, seuil confidence.
- KPIs : plaques détectées, images distinctes, confidence moyenne, bbox/image.
- Charts : prédiction, taille bbox, quality_flag, split, histogramme confidence.
- Bouton de rafraîchissement manuel + cache TTL 60 s.

---

## Schémas de données

### `DEST/` — sortie Producer (`destSchema`)

| Colonne | Type | Description |
|---------|------|-------------|
| `image_name` | String | Nom du fichier image (sans extension) |
| `split` | String | `train` / `valid` / `test` / `unknown` |
| `image_path` | String | Chemin complet de l'image source |
| `label` | String | Classe (YOLO classes.txt / COCO categories / VOC name) |
| `orig_width`, `orig_height` | Int | Dimensions originales de l'image |
| `image_data` | Binary | Image letterboxée 256×256 encodée JPEG |
| `class_id` | Int | Identifiant de classe (null si background) |
| `cx_norm`, `cy_norm`, `w_norm`, `h_norm` | Double | Bbox normalisée après letterbox |
| `x_min`, `y_min`, `x_max`, `y_max` | Double | Bbox en pixels (256×256) |
| `bbox_area`, `bbox_area_norm`, `aspect_ratio` | Double | Features géométriques |
| `bbox_size_category` | String | `small` / `medium` / `large` |
| `center_x_quadrant`, `center_y_quadrant` | String | `left`/`right`, `top`/`bottom` |
| `num_boxes` | Int | Nombre de plaques sur l'image |
| `ingested_at` | Timestamp | Timestamp d'ingestion Producer |

### `OUTPUT/` — sortie Consumer (`outputSchema` = destSchema +)

| Colonne | Type | Description |
|---------|------|-------------|
| `quality_flag` | String | `OK` / `TOO_SMALL` / `BAD_RATIO` / `NO_PLATE` |
| `prediction` | String | `plate` / `background` / `uncertain` |
| `confidence_score` | Double | Score de confiance [0, 1] |
| `processed_at` | Timestamp | Timestamp de traitement Consumer |

---

## Modèle ML — priorité de chargement (ModelLoader)

```
1. best_model.onnx    → PyTorch exporté (image-based : image_data [1,3,256,256])
                         produit par : ml/export_pytorch.py ← ml/checkpoints/best_model.pt
2. plate_model.onnx   → sklearn RandomForest (feature-based : 6 features bbox)
                         produit par : ml/train_model.py ← OUTPUT/
3. spark_model/       → Spark MLlib PipelineModel
4. (aucun fichier)    → scoring simulé heuristique (ratio bbox ≈ 3.1, area, quality_flag)
```

### Workflow modèle PyTorch (recommandé)

```bash
# 1. Exporter best_model.pt → ONNX
pip install torch onnx onnxruntime
python ml/export_pytorch.py \
  --checkpoint ml/checkpoints/best_model.pt \
  --output     ml/checkpoints/best_model.onnx \
  --img-size   256 \
  --num-classes 2

# 2. Copier vers HDFS (le Consumer le chargera au prochain démarrage)
hdfs dfs -put ml/checkpoints/best_model.onnx hdfs:///project/models/
```

> Si le chargement du modèle PyTorch échoue (architecture inconnue), `export_pytorch.py`
> tente un wrapper ResNet-18 avec `strict=False`. Définir manuellement l'architecture
> dans le script si nécessaire.

### Workflow modèle sklearn (alternatif, si best_model.onnx absent)

```bash
pip install pyspark scikit-learn skl2onnx onnxruntime
python ml/train_model.py --output-dir hdfs:///project/output/ --model-dir ./models
hdfs dfs -put models/plate_model.onnx hdfs:///project/models/
```

---

## Dépendances

| Composant | Version |
|-----------|---------|
| Scala | 2.13.14 |
| Spark | 4.1.1 |
| ONNX Runtime (Java) | 1.18.0 |
| Python | 3.8+ |
| scikit-learn | ≥ 1.3 |
| skl2onnx | ≥ 1.16 |
| onnxruntime (Python) | ≥ 1.18 |
| Streamlit | ≥ 1.30 |
