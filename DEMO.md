# Démo évaluateur — scénario pas à pas (~12 min)

## Avant l'arrivée de l'évaluateur (préparation, ~5 min)

```bash
# 1. Jars à jour
java -Xmx512m -jar sbt-launch.jar assembly

# 2. Modèle + métriques déjà en place (run hebdo fait la veille)
python main.py --weekly --skip-build        # ~10 min — à faire AVANT la démo

# 3. Préparer un lot d'images de démo dans un dossier à part
mkdir demo_images
# copier ~20 images de license-plate-detection-dataset-10125-images/test/images/
```

Vérifier : `data/models/plate_locator.onnx` + `metrics/*.json` présents.
Deux terminaux ouverts + un explorateur de fichiers sur
`data/source/continuous_ingress/`.

---

## Étape 1 — Architecture (1 min, ARCHITECTURE.md ouvert)

Montrer le diagramme et énoncer les deux temporalités **étanches** :
- **Flux continu** : ingress → Producer (éphémère) → DEST → Consumer
  (Structured Streaming permanent) → staging → PREDICT → output → dashboard.
- **Flux hebdo** : ml_dataset → copie batch → training Python → modèles ONNX
  + métriques versionnées.

Phrases clés attendues par l'évaluateur : *Hadoop FS exclusif côté Producer,
aucun prétraitement au transfert, Structured Streaming natif sans DStream ni
`while(true)`, ordre strict READ→PARSE→WRITE staging→PREDICT→WRITE output,
exactly-once par checkpoint + sink idempotent.*

## Étape 2 — Démarrer le flux continu (1 min)

```bash
python main.py --skip-build --producer-every 30
```

Montrer dans les logs :
- `[ModelLoader] Active mode: YOLO ONNX (plate_locator.onnx)` → le modèle est **broadcasté au démarrage** ;
- le dashboard s'ouvre sur http://localhost:8501 ;
- le Producer part puis **s'arrête tout seul** (`Terminé — arrêt du job`), et
  est relancé toutes les 30 s par l'orchestrateur (job éphémère + trigger).

## Étape 3 — La donnée traverse le pipeline en direct (3 min)

Glisser les ~20 images de `demo_images/` dans `data/source/continuous_ingress/`
(explorateur de fichiers = effet visuel).

Sur le dashboard, onglet **Suivi des prédictions & du flux** :
1. **En attente (SOURCE)** monte à 20 ;
2. au tick suivant du Producer : SOURCE retombe à 0 (les fichiers sont
   **déplacés**, pas copiés), **En transit (DEST)** monte ;
3. ~10-20 s après : **Traités (OUTPUT)** monte, la table de traçabilité passe
   les fichiers `Reçu → En cours → Traité` avec prédiction + confidence ;
4. la galerie affiche les images avec la **bbox prédite tracée en vert** ;
5. latence moyenne ingestion→prédiction affichée (~15 s).

## Étape 4 — Preuves d'ingénierie (3 min, terminal 2)

```powershell
# a) Sink idempotent + partitions par date (M1/C2)
ls data/output          # → ingest_date=YYYY-MM-DD/
ls data/output/ingest_date=*/            # → batch_id=N/ avec _SUCCESS

# b) DEST purgé après persistance (M2) — le listing du stream ne dégénère pas
ls data/dest/continuous_ingress          # → vide
ls data/archive/continuous_ingress      # → images archivées

# c) Crash test exactly-once (LE moment fort) :
#    tuer le consumer (Ctrl+C sur main.py), supprimer data/checkpoint,
#    relancer python main.py --skip-build → logs :
#    "[Consumer] batch 0 déjà présent ... — skip (rejeu)"  → 0 doublon
```

## Étape 5 — Flux hebdomadaire / MLOps (2 min)

Ne PAS relancer le training complet en live (long). Montrer :

```bash
# la commande (planifiable schtasks/cron) :
python main.py --weekly --skip-build
# et les artefacts du dernier run :
ls data/models                    # plate_locator.onnx (YOLOv8n)
cat data/models/metrics/v*.json   # IoU moyen/médian, % IoU >= 0.5 test/val
```

Dashboard onglet **Suivi de l'entraînement** : version du modèle actif, date
du run, volume du dataset, métriques IoU du détecteur, **courbe d'évolution
de l'IoU au fil des runs**.

Si l'évaluateur veut voir le training tourner : `python ml/train_yolo.py
--dataset-dir data/dest/ml_dataset --model-dir data/models --epochs 1` (~6 min).

## Étape 6 — Tests automatisés (1 min)

```bash
python -m pytest ml/tests -q          # 8 tests : IoU, labels, YAML dataset
java -Xmx512m -jar sbt-launch.jar test  # 6 tests : parité Py↔Scala, décodage
                                        # YOLO, idempotence rejeu, schémas
```

Souligner le test de **parité de prétraitement Python↔Scala** : le bug le plus
sournois d'un pipeline train-Python / score-Scala, verrouillé par fixture.

## Étape 7 — Questions probables (préparer les réponses)

| Question | Réponse courte |
|---|---|
| Pourquoi le Producer ne stream pas ? | Job batch éphémère par cahier des charges ; le streaming natif est le rôle du Consumer. |
| Exactly-once, vraiment ? | Checkpoint (source) + sink idempotent par `batch_id` + `_SUCCESS` (rejeu prouvé sans doublon). |
| Où le modèle est-il chargé ? | Broadcasté une fois au démarrage du stream ; une session ONNX par partition, fermée par `TaskCompletionListener`. |
| Pourquoi ONNX et pas PyTorch dans Spark ? | Scoring natif C++ dans la JVM sans UDF Python ni dépendance torch côté executors. |
| Le flux continu voit-il les labels ? | Jamais — labels uniquement dans ml_dataset (flux hebdo). La prédiction se fait sur les octets de l'image. |

## Plan B (pannes de démo)

- **Port 8501 occupé** : `python main.py --streamlit-only` après avoir tué
  l'ancien process ; ou changer de port dans main.py.
- **Spark RPC coupé (VPN/adaptateur virtuel)** : les confs
  `spark.driver.bindAddress=127.0.0.1` sont déjà dans main.py — lancer via
  main.py, pas spark-submit à la main.
- **Aucune donnée n'apparaît** : vérifier que les images sont bien des
  .jpg/.png (les autres fichiers sont filtrés), et que le Producer a fait son
  tick (`--producer-every 30`).
- **Démo offline totale** : `python main.py --streamlit-only` sur les données
  déjà traitées — tout le dashboard reste démontrable.
