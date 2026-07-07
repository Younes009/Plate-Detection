# Architecture — Plate Detection Spark Pipeline

## Vue d'ensemble : les deux temporalités

```
FLUX CONTINU (inférence, permanent)
SOURCE/continuous_ingress ──déplacement──▶ DEST/continuous_ingress ──stream──▶ staging ──▶ [PREDICT] ──▶ OUTPUT ──▶ Dashboard
                            Producer                                Consumer   (Parquet)    ONNX          (Parquet)   Streamlit
                            mode continuous                         Structured                ▲                       (READ 8)
                            (Hadoop FS, batch/interval)             Streaming                 │
                                                                                              │ EXPORT 6 (.pkl + .onnx)
FLUX HEBDOMADAIRE (réentraînement, planifié)                                                  │
SOURCE/ml_dataset ──copie récursive──▶ DEST/ml_dataset ──READ 5──▶ ML Python ────────────────┘
(train/test/val)    Producer            (arborescence     train_model.py
                    mode batch           préservée)       + métriques versionnées (MODEL/metrics/)
```

Les deux flux sont **étanches** : le moteur de streaming n'écoute QUE
`DEST/continuous_ingress/` ; la copie hebdo `DEST/ml_dataset/` n'entre jamais
dans le stream, elle est lue directement par le service ML Python.

---

## Zones de stockage

| Zone | Rôle |
|---|---|
| `data/source/continuous_ingress/` | Nouvelles images brutes arrivant au fil de l'eau (à prédire) |
| `data/source/ml_dataset/` | Données historiques d'entraînement — arborescence IMPÉRATIVE `{train,test,val(id)}/{images,labels}/` : chaque split contient de manière étanche ses images ET leurs labels YOLO `.txt` (le dataset livré joue ce rôle si le dossier est absent) |
| `data/dest/continuous_ingress/` | Images déplacées par le Producer, détectées par le Consumer |
| `data/dest/ml_dataset/` | Copie hebdomadaire intégrale, lue par le service ML |
| `data/staging/` | Couche intermédiaire propre (Parquet, une ligne/bbox) |
| `data/output/` | Résultats enrichis prediction + confidence (Parquet) |
| `data/models/` | `plate_model.pkl` (contrat) + `plate_model.onnx` (runtime Scala) + `metrics/*.json` (historique versionné) |

---

## Les 4 (micro)services

### 1. Producer — `producer/ProducerApp.scala` (Spark Core, Scala)

API **Hadoop FileSystem exclusivement** (`org.apache.hadoop.fs`), transfert
brut octet-à-octet, AUCUN prétraitement. **Job ÉPHÉMÈRE dans les deux modes**
— il transfère puis s'arrête tout seul, c'est un déclencheur externe qui le
relance (main.py toutes les `--producer-every` secondes, ou cron/trigger).
Paramètres : `sourcePath destPath batchSize intervalSeconds recursive mode`.

- **mode `continuous`** (flux continu, relancé régulièrement) : scanne
  `SOURCE/continuous_ingress/` (images brutes UNIQUEMENT — aucun label YOLO
  ici) et **déplace** tout vers `DEST/continuous_ingress/`, par lots de
  `batchSize`, puis exit. Le déplacement vide la source.
- **mode `batch`** (flux hebdo, lancé une fois par semaine) : copie
  **récursive et intégrale** de `SOURCE/ml_dataset/` vers `DEST/ml_dataset/`
  en préservant l'arborescence `train/test/val` (images ET labels), lot par
  lot, puis exit. Idempotent (rejouable).

### 2. Consumer & Predictor — `consumer/ConsumerApp.scala` (Spark Structured Streaming, Scala)

Structured Streaming **natif** (pas de DStream, pas de boucle manuelle) ;
écoute permanente (`awaitTermination`, checkpoint exactly-once). Ordre strict
dans chaque micro-batch (`foreachBatch`) :

1. **FETCH** : `readStream.format("binaryFile")` sur `DEST/continuous_ingress/`
   uniquement (`recursiveFileLookup`, `maxFilesPerTrigger=100`, trigger 10 s).
2. **PARSE & TRANSFORM** : filtre des images valides, typage
   (`image_name, image_path, image_data, length, ingested_at, processed_at`).
   **Aucune recherche de label** : le flux ne transporte que des images brutes.
3. **WRITE INTERMÉDIAIRE** : écriture immédiate des images propres dans
   `data/staging/` (Parquet Snappy) — couche d'historique.
4. **INJECTION DU MODÈLE & PREDICT** : `ModelLoader.score()` — modèle de
   vision ONNX broadcasté (injecté au démarrage), scoring **directement sur
   les octets de l'image**, ajoute `prediction` + `confidence_score`.
5. **WRITE OUTPUT** : résultats enrichis dans `data/output/`.

### 3. ML Training — `ml/train_model.py` (Python, indépendant de Spark)

Flux **hebdomadaire** (planifiable via `schtasks` / `cron` sur
`python main.py --weekly`) :

- **READ** : lit la copie `DEST/ml_dataset/` (images + labels YOLO,
  splits `train/test/val(id)` depuis l'arborescence). Les labels ne servent
  QUE là — jamais dans le stream.
- Features = pixels réduits (S×S RGB, aplati CHW, /255 — S=32 par défaut),
  EXACTEMENT le prétraitement de `ModelLoader.scala` au scoring ; cible =
  présence d'un label YOLO non vide.
- **EXPORT** : `plate_model.onnx` (entrée `[None, 3*S*S]` — le Consumer déduit
  S de la forme du modèle, opset 21) + `plate_model.pkl` (backup) dans
  `data/models/`.
- **Historique** : accuracy / F1 / matrice de confusion (test + val), volume
  du dataset et version du run dans `data/models/metrics/<version>.json`.

### 4. Dashboard — `streamlit/app.py` (Streamlit, Python)

**READ 8** : lecture continue de l'état du pipeline (auto-refresh non bloquant
via `st.fragment(run_every=...)`, cache TTL 30 s), sans jamais charger les
octets images depuis les Parquet. **Deux onglets** :

- **Suivi des prédictions & du flux** : Pipeline Tracker `st.columns`
  (En attente `/SOURCE` → En transit `/DEST` → Traités `/OUTPUT`), table de
  traçabilité par fichier (statut Reçu / En cours / Traité, horodatage,
  prédiction + confidence), distribution des prédictions, débit du stream,
  aperçu des dernières images (bbox dessinées).
- **Suivi de l'entraînement** : modèle actif (version, date du run hebdo,
  volume dataset, artefacts), accuracy / F1 / matrices de confusion test &
  val du dernier run, importance des features, évolution historique de la
  précision au fil des semaines.

---

## Chargement du modèle — `ml/ModelLoader.scala`

Priorité (première correspondance gagne) — scoring TOUJOURS sur les octets image :

1. `best_model.onnx` — CNN PyTorch (entrée `[1,3,256,256]`, ImageNet normalized)
2. `plate_model.onnx` — RandomForest sklearn sur pixels réduits (entrée
   `[None, 3*S*S]` — S déduit de la forme du modèle, /255, CHW)
3. Fallback neutre (`prediction=uncertain`, `confidence=0.5`)

Le fichier ONNX est broadcasté aux executors ; une session ONNX Runtime est
ouverte par partition puis fermée après consommation. Les sorties
`label`/`probabilities` (skl2onnx récent) et `output_label`/`output_probability`
(anciennes versions) sont toutes deux acceptées.

---

## Schémas des données

### DEST — fichiers bruts (source streaming `binaryFile`, schéma imposé par Spark)

| Colonne | Type | Description |
|---|---|---|
| `path` | String | URI du fichier |
| `modificationTime` | Timestamp | Date de dépôt |
| `length` | Long | Taille en octets |
| `content` | Binary | Contenu brut |

### STAGING — écrit par le Consumer (WRITE intermédiaire), une ligne par image

| Colonne | Type | Description |
|---|---|---|
| `image_name` | String | Nom du fichier sans extension |
| `image_path` | String | URI de l'image |
| `image_data` | Binary | Contenu brut JPEG/PNG |
| `length` | Long | Taille en octets |
| `ingested_at` | Timestamp | Date de dépôt dans DEST |
| `processed_at` | Timestamp | Date de traitement |

Pas de colonnes label/bbox : le flux d'inférence ne transporte que des images.

### OUTPUT — staging + scoring

| Colonne | Type | Description |
|---|---|---|
| `prediction` | String | `plate` / `background` / `uncertain` |
| `confidence_score` | Double | Score `[0.0, 1.0]` |

### MODEL/metrics/<version>.json — historique d'entraînement

`version`, `trained_at`, `dataset_images`, `split_counts`,
`feature_importances`, `metrics.{test,val}.{accuracy, f1_score, confusion_matrix, n_samples}`.

---

## Stack technique

| Composant | Technologie | Version |
|---|---|---|
| Producer / Consumer | Scala + Apache Spark | 2.13.14 / **4.1.1** |
| Build | sbt + sbt-assembly | 1.9.9 |
| Runtime JVM | Java | 17 |
| ML scoring | ONNX Runtime (Java) | 1.20 |
| ML training | scikit-learn + skl2onnx (opset 21) | — |
| Dashboard | Streamlit + pandas + pyarrow + altair + pillow | ≥ 1.49 |

---

## Lancement

```bash
# Flux continu complet (producer continuous + consumer + dashboard)
python main.py

# Simuler des arrivées puis lancer le flux continu
python main.py --seed 50 --skip-build

# Flux hebdomadaire : copie ml_dataset + réentraînement + métriques
python main.py --weekly --skip-build

# Dashboard seul
python main.py --streamlit-only

# Producer seul : source dest batchSize intervalSeconds recursive mode
spark-submit --master local[2] producer/target/scala-2.13/producer-assembly-0.1.0.jar \
  data/source/continuous_ingress data/dest/continuous_ingress 100 5 true continuous

# Consumer seul : ingress staging output models checkpoint
spark-submit --master local[2] consumer/target/scala-2.13/consumer-assembly-0.1.0.jar \
  data/dest/continuous_ingress data/staging data/output data/models data/checkpoint

# Entraînement seul (après un passage du Producer batch)
python ml/train_model.py --dataset-dir data/dest/ml_dataset --model-dir data/models
```

Planification hebdomadaire :
- **Windows** : `schtasks /Create /SC WEEKLY /D MON /ST 03:00 /TN PlateRetrain /TR "python C:\...\main.py --weekly --skip-build"`
- **Linux** : `0 3 * * 1  cd /opt/pipeline && python main.py --weekly --skip-build`
