# License Plate Detection — Pipeline Spark Streaming + ML

Pipeline Big Data à deux temporalités (voir [ARCHITECTURE.md](ARCHITECTURE.md)) :

- **Flux continu (inférence)** : Producer **éphémère** (Hadoop FS, relancé
  périodiquement par un trigger) déplace les images brutes — sans labels —
  `SOURCE/continuous_ingress/` → `DEST/continuous_ingress/` ; Consumer
  (Spark Structured Streaming, **seul service permanent**) les capte, écrit la
  couche staging, applique le modèle de vision ONNX **directement sur l'image**
  et écrit `OUTPUT/` ; dashboard Streamlit en lecture continue.
- **Flux hebdomadaire (réentraînement)** : Producer en mode batch copie
  l'intégralité de `SOURCE/ml_dataset/` (arborescence `train/test/val`, images
  + labels YOLO) vers `DEST/ml_dataset/` ; le service ML Python fine-tune
  YOLOv8n, exporte `plate_locator.onnx` dans `MODEL/` et versionne ses
  métriques IoU.

---

## Structure du projet

```
plate-detection-spark/
├── main.py                                                           # orchestrateur des deux flux
├── build.sbt                                                         # Spark 4.1.1, onnxruntime 1.20
├── sbt-launch.jar                                                    # launcher sbt embarqué
├── common/src/main/scala/com/project/common/Utils.scala              # schémas binaryFile/staging/output
├── producer/src/main/scala/com/project/producer/ProducerApp.scala    # Microservice 1 — modes continuous & batch
├── consumer/src/main/scala/com/project/consumer/ConsumerApp.scala    # Microservice 2 — Structured Streaming
├── ml/
│   ├── src/main/scala/com/project/ml/ModelLoader.scala               # scoring ONNX broadcasté
│   ├── train_yolo.py                                                 # détecteur YOLOv8n → plate_locator.onnx + métriques
│   ├── tests/                                                        # pytest (IoU, prétraitement, labels)
│   └── requirements.txt
├── streamlit/
│   ├── app.py                                                        # dashboard 2 onglets (flux + entraînement)
│   └── requirements.txt
└── data/                                                             # zones de stockage (créées par main.py)
    ├── source/{continuous_ingress, ml_dataset}/
    ├── dest/{continuous_ingress, ml_dataset}/
    ├── staging/  ├── output/  ├── models/{metrics/}  └── checkpoint/
```

> Si `data/source/ml_dataset/` est absent, le dataset livré
> `license-plate-detection-dataset-10125-images/` joue ce rôle automatiquement.
>
> Arborescence impérative de `ml_dataset/` — chaque split est étanche
> (ses labels ne concernent que ses propres images) :
>
> ```
> ml_dataset/
> ├── train/ ├── images/*.jpg
> │          └── labels/*.txt
> ├── test/  ├── images/*.jpg
> │          └── labels/*.txt
> └── val/   ├── images/*.jpg      # "valid" accepté aussi
>            └── labels/*.txt
> ```

---

## Prérequis

- Java 17, Spark **4.1.1** (`spark-submit` dans le PATH — la version doit
  matcher `build.sbt`), Python 3.11+
- `pip install -r ml/requirements.txt -r streamlit/requirements.txt`

---

## Lancement rapide — `main.py`

```bash
# Flux continu complet : build → producer continuous → consumer → dashboard
python main.py

# Simuler des arrivées (50 images test → SOURCE/continuous_ingress) puis flux continu
python main.py --seed 50 --skip-build

# Flux hebdomadaire : copie ml_dataset + réentraînement + métriques, puis exit
python main.py --weekly --skip-build

# Dashboard seul
python main.py --streamlit-only
```

**Options :**

| Option | Défaut | Description |
|--------|--------|-------------|
| `--weekly` | false | Run hebdo : Producer batch + entraînement ML, puis exit |
| `--seed N` | 0 | Dépose N images test dans `SOURCE/continuous_ingress` (simulation) |
| `--skip-build` | false | Skip `sbt assembly` |
| `--streamlit-only` | false | Lance uniquement le dashboard |
| `--batch-size N` | 100 | Fichiers par lot Producer |
| `--interval N` | 0 | Secondes entre lots Producer |
| `--producer-every S` | 60 | Période de relance du Producer continuous (éphémère) |

**Séquence du flux continu :** consumer streaming permanent (arrière-plan) →
producer éphémère relancé toutes les `--producer-every` secondes par
l'orchestrateur (rôle de trigger) → Streamlit en foreground sur `:8501` →
`Ctrl+C` arrête tout.

**Planification hebdomadaire :**

```bash
# Windows
schtasks /Create /SC WEEKLY /D MON /ST 03:00 /TN PlateRetrain /TR "python C:\...\main.py --weekly --skip-build"
# Linux (crontab)
0 3 * * 1  cd /opt/pipeline && python main.py --weekly --skip-build
```

---

## Build manuel

```bash
java -Xmx512m -jar sbt-launch.jar assembly
```

Produit `producer-assembly-0.1.0.jar` et `consumer-assembly-0.1.0.jar`
(`onnxruntime 1.20` bundlé dans le fat-jar du Consumer).

---

## Exécution manuelle des services

### Microservice 1 — Producer (Hadoop FS, transfert brut)

```bash
# Args : sourcePath destPath batchSize intervalSeconds recursive mode
# Job ÉPHÉMÈRE dans les deux modes : transfère puis s'arrête tout seul.

# Flux continu : un cycle de déplacement (à relancer via cron/trigger/main.py)
spark-submit --master local[2] \
  producer/target/scala-2.13/producer-assembly-0.1.0.jar \
  data/source/continuous_ingress data/dest/continuous_ingress 100 0 true continuous

# Flux hebdo : copie récursive intégrale (arborescence train/test/val préservée)
spark-submit --master local[2] \
  producer/target/scala-2.13/producer-assembly-0.1.0.jar \
  data/source/ml_dataset data/dest/ml_dataset 100 0 true batch
```

### Microservice 2 — Consumer (Structured Streaming)

```bash
# Args : ingressPath stagingPath outputPath modelPath checkpointPath
spark-submit --master local[2] \
  consumer/target/scala-2.13/consumer-assembly-0.1.0.jar \
  data/dest/continuous_ingress data/staging data/output data/models data/checkpoint
```

Ordre strict par micro-batch : **FETCH → PARSE/TRANSFORM → WRITE staging →
PREDICT (vision ONNX, directement sur les octets image — aucun label dans le
stream) → WRITE output**. Exactly-once via checkpoint, `awaitTermination()` —
aucun DStream, aucune boucle manuelle. Seul service permanent du pipeline.

### Service ML — réentraînement hebdomadaire

```bash
# Fine-tune YOLOv8n (IoU test ~0.77) + éval IoU + export ONNX + métriques
python ml/train_yolo.py --dataset-dir data/dest/ml_dataset --model-dir data/models --epochs 3
```

- Lit la copie hebdo (images + labels YOLO, splits depuis l'arborescence) —
  les labels ne servent qu'ici, jamais dans le stream.
- Exporte `plate_locator.onnx` (entrée `[1,3,256,256]`, opset 17 — requis par
  onnxruntime Java 1.20).
- Versionne IoU moyen/médian, % IoU ≥ 0.5 (test + val) dans
  `data/models/metrics/<version>.json`.
- Le Consumer charge le nouveau modèle à son prochain démarrage.

### Dashboard

```bash
python -m streamlit run streamlit/app.py
```

Deux onglets :
1. **Suivi des prédictions & du flux** — Pipeline Tracker (En attente `/SOURCE`
   → En transit `/DEST` → Traités `/OUTPUT`), table de traçabilité par fichier
   (Reçu / En cours / Traité, prédiction + confidence), débit du stream,
   aperçu des dernières images avec bbox.
2. **Suivi de l'entraînement** — modèle actif (version, date, volume dataset),
   métriques du dernier run (accuracy, F1, matrices de confusion test/val),
   évolution historique de la précision.

---

## Schémas de données

Voir [ARCHITECTURE.md](ARCHITECTURE.md#schémas-des-données). En résumé :

- **DEST** : schéma fixe de la source `binaryFile` (`path`, `modificationTime`,
  `length`, `content`) — fichiers bruts, aucun prétraitement.
- **STAGING** : une ligne par image propre — `image_name`, `image_path`,
  `image_data`, `length`, `ingested_at`, `processed_at` (aucun label/bbox :
  le flux d'inférence ne transporte que des images).
- **OUTPUT** : staging + `prediction` (`plate`/`background`/`uncertain`) +
  `confidence_score` [0, 1].
- **MODEL/metrics/*.json** : version, date, volume, accuracy/F1/confusion par split.

---

## Modèle ML — priorité de chargement (ModelLoader)

```
1. plate_locator.onnx → détecteur YOLOv8n [1,3,S,S] → [1,4+k,N]
                        (S déduit de la forme du modèle ; prediction,
                        confidence ET bbox = meilleure détection)
2. (aucun fichier)    → fallback neutre (uncertain, 0.5)
```

Scoring TOUJOURS directement sur les octets de l'image — aucun label requis.

Le fichier ONNX est broadcasté aux executors ; une session ONNX Runtime par
partition. Les noms de sorties skl2onnx anciens (`output_label`/`output_probability`)
et récents (`label`/`probabilities`) sont tous deux acceptés.

---

## Dépendances

| Composant | Version |
|-----------|---------|
| Scala | 2.13.14 |
| Spark | **4.1.1** (runtime = build, sinon `NoSuchMethodError`) |
| ONNX Runtime (Java) | 1.20.0 |
| Python | 3.11+ |
| scikit-learn / skl2onnx | ≥ 1.4 / ≥ 1.16 (export opset 21) |
| Streamlit | ≥ 1.49 |
