Tu es un expert Apache Spark (Scala) spécialisé en architectures Big Data industrielles.

Je veux que tu crées un projet complet Spark en Scala avec architecture batch + streaming basé sur HDFS.

OBJECTIF:
Construire un pipeline Big Data avec :
- un Producer batch (Spark Core + Scala)
- un Consumer streaming (Spark Structured Streaming + Scala)

ARCHITECTURE OBLIGATOIRE:

1. PRODUCER (Batch - Spark Core + Scala)
- Lire un dataset (CSV ou données simulées)
- Nettoyer et transformer les données
- Écrire UNE SEULE FOIS dans HDFS en Parquet
- Path: hdfs:///project/raw/
- Doit s’arrêter après exécution
- Aucun streaming, aucune boucle

2. CONSUMER (Streaming - Spark Structured Streaming + Scala)
- Lire en continu depuis: hdfs:///project/raw/
- Utiliser obligatoirement readStream
- Appliquer des transformations (filter, enrichissement, agrégations optionnelles)
- Écrire vers: hdfs:///project/processed/
- Utiliser checkpointing: hdfs:///project/checkpoint/
- Doit tourner en continu avec awaitTermination()

HDFS STRUCTURE:
- hdfs:///project/raw/
- hdfs:///project/processed/
- hdfs:///project/staging/
- hdfs:///project/logs/
- hdfs:///project/checkpoint/

CONTRAINTES:
- Scala 2.12+
- Spark 3.x
- DataFrame API uniquement (pas de RDD)
- Parquet obligatoire
- Producer = batch uniquement (Spark Core + Scala)
- Consumer = streaming uniquement (Spark Structured Streaming)
- HDFS uniquement (aucun fichier local)

OPTION BONUS:
- Ajouter un module de scoring ML simple (Spark MLlib ou colonne simulée)

STRUCTURE DU PROJET:
- /producer/src/main/scala/ProducerApp.scala
- /consumer/src/main/scala/ConsumerApp.scala
- /common/Utils.scala (optionnel)
- build.sbt

OUTPUT ATTENDU:
1. Code Scala complet (Producer + Consumer + optional Utils)
2. build.sbt complet avec dépendances Spark
3. Explication rapide du flux de données

IMPORTANT:
- Producer doit s’exécuter une seule fois et s’arrêter
- Consumer doit tourner indéfiniment
- Consumer doit utiliser Structured Streaming (readStream obligatoire)

le dataset est license-plate-detection-dataset-10125-images
sous 3 dossiers et chaque dossiers continet images labels Image Formats: JPEG / PNG (high-resolution)

Annotation Format: Bounding boxes for license plates

Supported Export Formats: YOLOv11, YOLOv10, YOLOv9, YOLOv8, YOLOv7, COCO JSON, Pascal VOC XML, TFRecord, CreateML JSON, and others
- 