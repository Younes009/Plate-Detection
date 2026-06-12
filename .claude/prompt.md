Act en tant que Principal Data Engineer et Architecte Big Data Senior (niveau d'exigence : Uber, Netflix, Airbnb). Tu dois concevoir l'intégralité d'un pipeline de production Big Data pour la détection de plaques d'immatriculation à partir du jeu de données "license-plate-detection-dataset-10125-images" (contenant des images JPEG/PNG haute résolution et des bounding boxes aux formats YOLO/COCO/Pascal VOC).

Génère un projet COMPLET, EXÉCUTABLE, SANS AUCUN RACCOURCI ni "TODO", respectant scrupuleusement les exigences industrielles ci-dessous.

────────────────────────────────────────────────────────────────────────────────
1. ARCHITECTURE ET ARBORESCENCE DU PROJET
   ────────────────────────────────────────────────────────────────────────────────
   Génère l'arborescence exacte et crée l'intégralité du code pour chacun des fichiers suivants :
   /build.sbt
   /common/src/main/scala/com/project/common/Utils.scala
   /producer/src/main/scala/com/project/producer/ProducerApp.scala
   /consumer/src/main/scala/com/project/consumer/ConsumerApp.scala
   /ml/src/main/scala/com/project/ml/ModelLoader.scala
   /streamlit/app.py

────────────────────────────────────────────────────────────────────────────────
2. CONFIGURATION BUILD (build.sbt)
   ────────────────────────────────────────────────────────────────────────────────
   Fournis un fichier `build.sbt` multi-modules complet pour Scala 2.12.18+ et Spark 3.5.x.
- Modules : `common`, `producer`, `consumer`, `ml`.
- Incluts explicitement toutes les dépendances nécessaires : `spark-core`, `spark-sql`, `spark-mllib`, ainsi que les connecteurs natifs ou librairies de parsing JSON/XML si nécessaires pour COCO/Pascal VOC.
- Configure une stratégie de merge propre (`assemblyMergeStrategy`) pour éviter les conflits lors de la génération des Fat Jars.

────────────────────────────────────────────────────────────────────────────────
3. MODULE PRODUCER : INGESTION BATCH STRICTE
   ────────────────────────────────────────────────────────────────────────────────
   Fichier : `ProducerApp.scala`
- Paradigme : Batch Pur (Exécution unique, arrêt automatique après traitement, interdiction d'utiliser des boucles infinies ou du streaming).
- Logique métier :
    1. Parcourir récursivement le dossier du dataset d'images et ses fichiers d'annotations (YOLO/COCO/Pascal VOC).
    2. Parser les annotations pour extraire : `image_path`, `label`, `bbox` (coordonnées normalisées ou absolues), et `class`.
    3. Structurer ces données dans un DataFrame Spark natif fortement typé (sans passer par les RDDs bas niveau).
    4. Appliquer un nettoyage rigoureux (gestion des valeurs manquantes, validation des chemins d'images).
- Écriture : Écrire le DataFrame final au format Parquet optimisé (avec compression Snappy) à l'emplacement exact : `hdfs:///project/raw/`.

────────────────────────────────────────────────────────────────────────────────
4. MODULE CONSUMER : STRUCTURED STREAMING & ML DOUBLE FLUX
   ────────────────────────────────────────────────────────────────────────────────
   Fichier : `ConsumerApp.scala`
   C'est le cœur du système. Tu dois implémenter une architecture de double lecture et double écriture en streaming continu :

- FLUX DE LECTURE N°1 (Data Stream) :
  Utilise obligatoirement `spark.readStream` au format Parquet sur `hdfs:///project/raw/`. Configure l'inférence de schéma explicite (`schema`) pour garantir la stabilité du flux.

- FLUX DE LECTURE N°2 (Modèle ML) :
  Charge un modèle d'inférence (via `ModelLoader.scala`). Ce modèle peut être un modèle Spark MLlib (`PipelineModel` ou `RandomForestClassificationModel`) préalablement entraîné, ou un modèle d'inférence simulé de manière très réaliste (Fallback d'évaluation orienté computer vision avec score de confiance et prédiction de classe).

- LOGIQUE DE MICRO-BATCH ET PROCESSING :
  Pour chaque micro-batch, applique un filtrage des anomalies, du feature engineering de production, puis injecte le modèle ML pour ajouter deux colonnes calculées au flux : `prediction` (string/int) et `confidence_score` (double).

- DOUBLE ÉCRITURE SÉCURISÉE :
    1. Écriture principale : Enregistrer le flux enrichi (features, labels, metadata, prédictions) dans `hdfs:///project/processed/` en mode `append`.
    2. Écriture dédiée aux prédictions : Enregistrer de manière isolée un sous-ensemble critique (`image_path`, `prediction`, `confidence_score`) dans `hdfs:///project/processed/predictions/`.

- TOLÉRANCE AUX PANNES :
  Configure obligatoirement un mécanisme de Checkpointing strict sur `hdfs:///project/checkpoint/` pour garantir la sémantique "Exactly-Once". Le flux doit se terminer par `awaitTermination()` pour tourner indéfiniment.

────────────────────────────────────────────────────────────────────────────────
5. INTERFACE UTILISATEUR (Streamlit)
   ────────────────────────────────────────────────────────────────────────────────
   Fichier : `app.py`
   Développe une application Streamlit complète et élégante.
- Contrainte absolue : Lecture seule (`READ ONLY`). L'application ne doit effectuer aucun calcul de Machine Learning ni aucune transformation lourde.
- Logique : Elle doit se connecter via un client compatible (ou simuler proprement l'accès aux répertoires HDFS) pour lire directement `hdfs:///project/processed/` et `hdfs:///project/processed/predictions/`.
- Fonctionnalités graphiques :
    - Rafraîchissement automatique ou bouton de synchronisation des données du stream.
    - Dashboards de KPIs majeurs (nombre de plaques détectées, taux de confiance moyen).
    - Graphiques et histogrammes dynamiques de la distribution des scores de confiance.
    - Table de données filtrable par classe ou par seuil de confiance.

────────────────────────────────────────────────────────────────────────────────
6. EXPLICATIONS ET DEPLOYMENT GUIDE
   ────────────────────────────────────────────────────────────────────────────────
   En fin de réponse, fournis un guide d'exécution concis décrivant :
1. La structure de dossiers requise au sein de HDFS (`/raw`, `/processed`, `/processed/predictions`, `/staging`, `/logs`, `/checkpoint`).
2. Les commandes `spark-submit` exactes pour lancer le Producer (Batch) puis le Consumer (Streaming) avec les configurations mémoire d'usage en production.

Ne fais aucune concession sur la syntaxe Scala ou Python. Fournis un code propre, modulaire et directement compilable.