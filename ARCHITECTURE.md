# Architecture — Plate Detection Spark Pipeline

## Vue d'ensemble

```
Dataset ──copie brute──▶ data/dest/ ──stream──▶ data/staging/ ──▶ [PREDICT] ──▶ data/output/ ──▶ Dashboard
(SOURCE)   Producer       (fichiers    Consumer   (Parquet,          modèle       (Parquet,       Streamlit
           Hadoop FS       bruts)      Structured  features)         ONNX          + prédictions)  (READ 8)
           READ 1 /        WRITE 2     Streaming   WRITE 4             ▲
           batch/interval              FETCH 3                          │ EXPORT 6 (.onnx / .pkl)
                                                       │                │
                                                       └── READ 5 ──▶ ML Python (train/test/val)
```

Numérotation identique au schéma `archi_structured_streaming.jpg`.

---

## Les 4 (micro)services

### 1. Producer — `producer/ProducerApp.scala` (Spark Core, Scala)

- **READ 1** : liste les images du dataset SOURCE via `org.apache.hadoop.fs.FileSystem`
  (récursivité paramétrable).
- **WRITE 2** : copie **brute, octet-à-octet** (`FileUtil.copy`) vers `data/dest/`,
  en conservant l'arborescence relative (`train/images/x.jpg`, `train/labels/x.txt`).
  **Aucun prétraitement, aucune conversion.** Le label YOLO est copié avant son image.
- Paramètres : `sourcePath destPath batchSize intervalSeconds recursive`.
- Envoie lot par lot, dort `intervalSeconds` entre deux lots, s'arrête à la fin.
- Idempotent : les fichiers déjà présents dans DEST sont ignorés (rejouable).

### 2. Consumer & Predictor — `consumer/ConsumerApp.scala` (Spark Structured Streaming, Scala)

Écoute permanente (`awaitTermination`, checkpoint exactly-once). Ordre strict
dans chaque micro-batch (`foreachBatch`) :

1. **FETCH 3** : `readStream.format("binaryFile")` sur `data/dest/`
   (`recursiveFileLookup`, `maxFilesPerTrigger=100`, trigger 10 s).
2. **PARSE & TRANSFORM** : extraction `image_name` / `split`, lecture du label
   YOLO via Hadoop FS, explode 1 ligne par bounding box, features géométriques
   typées (`class_id, cx_norm, cy_norm, w_norm, h_norm, bbox_area_norm, aspect_ratio`).
3. **WRITE 4** : écriture immédiate des données propres dans `data/staging/`
   (Parquet Snappy) — c'est la couche lue par le service ML.
4. **PREDICT 7** : `ModelLoader.score()` — modèle ONNX broadcasté (injecté au
   démarrage), ajoute `prediction` + `confidence_score`.
5. **WRITE OUTPUT** : résultats enrichis dans `data/output/`.

### 3. ML Training — `ml/train_model.py` (Python, indépendant de Spark)

- **READ 5** : lit `data/staging/` (pandas + pyarrow, colonnes features uniquement).
- Train/Val/Test selon la colonne `split` ; cible = présence d'une bbox YOLO.
- **EXPORT 6** : `plate_model.onnx` (consommé par le Consumer via ONNX Runtime)
  et `plate_model.pkl` (backup) dans `data/models/`.

### 4. Dashboard — `streamlit/app.py` (Streamlit, Python)

- **READ 8** : lit `data/output/` en continu (auto-refresh non bloquant via
  `st.fragment(run_every=...)`, cache TTL 30 s), sans jamais charger les
  octets images depuis les Parquet.
- 4 onglets : **Vue d'ensemble** (KPIs, prédictions, splits, débit du stream),
  **Analyse bbox** (histogrammes aire/ratio, scatter des centres),
  **Galerie** (images lues sur disque via `image_path`, bbox ground truth
  dessinées), **Données** (table filtrable + export CSV).
- Filtres sidebar : split, prédiction, confidence minimum, labels seulement.

---

## Chargement du modèle — `ml/ModelLoader.scala`

Priorité (première correspondance gagne) :

1. `best_model.onnx` — CNN PyTorch exporté (`ml/export_pytorch.py`), scoring sur `image_data`
2. `plate_model.onnx` — RandomForest sklearn (`ml/train_model.py`), scoring sur les 6 features bbox
3. `spark_model/` — PipelineModel Spark MLlib
4. Fallback heuristique (`prediction=uncertain`, `confidence=0.5`)

Le fichier ONNX est broadcasté aux executors ; une session ONNX Runtime est
ouverte par partition puis fermée après consommation.

---

## Schémas des données

### DEST — fichiers bruts (source streaming `binaryFile`, schéma imposé par Spark)

| Colonne | Type | Description |
|---|---|---|
| `path` | String | URI du fichier |
| `modificationTime` | Timestamp | Date de dépôt |
| `length` | Long | Taille en octets |
| `content` | Binary | Contenu brut |

### STAGING — écrit par le Consumer (WRITE 4), une ligne par bbox

| Colonne | Type | Description |
|---|---|---|
| `image_name` | String | Nom du fichier sans extension |
| `split` | String | `train` / `valid` / `test` / `unknown` |
| `image_path` | String | URI de l'image |
| `image_data` | Binary | Contenu brut JPEG/PNG |
| `ingested_at` | Timestamp | Date de dépôt dans DEST |
| `label_path` / `label_data` | String | URI et contenu YOLO brut |
| `class_id` | Int | Classe YOLO (null si pas de label) |
| `cx_norm, cy_norm, w_norm, h_norm` | Double | Bbox YOLO normalisée |
| `bbox_area_norm`, `aspect_ratio` | Double | Features dérivées |
| `processed_at` | Timestamp | Date de traitement |

### OUTPUT — staging + scoring

| Colonne | Type | Description |
|---|---|---|
| `prediction` | String | `plate` / `background` / `uncertain` |
| `confidence_score` | Double | Score `[0.0, 1.0]` |

---

## Stack technique

| Composant | Technologie | Version |
|---|---|---|
| Producer / Consumer | Scala + Apache Spark | 2.13.14 / **4.1.1** |
| Build | sbt + sbt-assembly | 1.9.9 |
| Runtime JVM | Java | 17 |
| ML scoring | ONNX Runtime | 1.18 |
| ML training | scikit-learn + skl2onnx | — |
| Dashboard | Streamlit + pandas + pyarrow + altair + pillow | ≥ 1.49 |

---

## Lancement

```bash
# Pipeline complet
python main.py

# Sans rebuild
python main.py --skip-build

# Producer seul : source dest batchSize intervalSeconds recursive
spark-submit --master local[2] \
  producer/target/scala-2.13/producer-assembly-0.1.0.jar \
  license-plate-detection-dataset-10125-images \
  data/dest 200 5 true

# Consumer seul : dest staging output models checkpoint
spark-submit --master local[2] \
  consumer/target/scala-2.13/consumer-assembly-0.1.0.jar \
  data/dest data/staging data/output data/models data/checkpoint

# Entraînement ML (après un premier passage du Consumer)
python ml/train_model.py --staging-dir data/staging --model-dir data/models

# Dashboard seul
python main.py --streamlit-only
```
