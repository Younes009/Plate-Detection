Tu es un expert Apache Spark (Scala) spécialisé en architectures Big Data industrielles (niveau production : Uber / Netflix / Airbnb).

Tu dois générer un projet COMPLET, EXÉCUTABLE et STRICTEMENT conforme.

────────────────────────────────────────
OBJECTIF GLOBAL
────────────────────────────────────────

Construire un pipeline Big Data complet basé sur :

Dataset:
license-plate-detection-dataset-10125-images

Contenu:
- images JPEG/PNG haute résolution
- annotations bounding boxes
- formats: YOLO / COCO / Pascal VOC

Pipeline:
Batch ingestion → HDFS RAW → Streaming processing + ML → HDFS PROCESSED → Streamlit

────────────────────────────────────────
ARCHITECTURE VISUELLE OBLIGATOIRE
────────────────────────────────────────

Respect EXACT du flux :

\documentclass[tikz,border=10mm]{standalone}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}
\usepackage{xcolor}

\definecolor{sparkgreen}{HTML}{7FFF00}
\definecolor{pythonorange}{HTML}{DEB887}
\definecolor{predictpurple}{HTML}{DA70D6}
\definecolor{streamlitblue}{HTML}{87CEEB}

\begin{document}
\begin{tikzpicture}

\node (source) {Dataset Images + Labels};
\node (producer) [right=of source] {PRODUCER (Spark Core Scala BATCH)};
\node (raw) [right=of producer] {hdfs:///project/raw/};

\node (consumer) [below=of raw] {CONSUMER (Structured Streaming + ML)};
\node (processed) [right=of consumer] {hdfs:///project/processed/};

\node (streamlit) [below=of processed] {STREAMLIT (READ ONLY)};

\draw[->] (source) -- (producer) node[midway,above]{READ};
\draw[->] (producer) -- (raw) node[midway,above]{WRITE PARQUET};

\draw[->] (raw) -- (consumer) node[midway,right]{READ STREAM (DATA)};

\draw[->] (consumer) -- (processed) node[midway,above]{WRITE PREDICTIONS};

\draw[->] (processed) -- (streamlit) node[midway,right]{READ ONLY};

\end{tikzpicture}
\end{document}

────────────────────────────────────────
ARCHITECTURE PRODUCER (BATCH ONLY)
────────────────────────────────────────

- Spark Core + Scala uniquement
- Lit dataset images + annotations
- Parse YOLO / COCO
- Nettoyage + structuration

Sortie DataFrame :
(image_path, label, bbox, class)

Écriture UNIQUE :
hdfs:///project/raw/ (Parquet)

STOP après écriture
AUCUN streaming
AUCUNE boucle infinie

────────────────────────────────────────
ARCHITECTURE CONSUMER (TRÈS IMPORTANT)
────────────────────────────────────────

Le consumer fait DOUBLE LECTURE + DOUBLE USAGE :

────────────────────────────
1. READ #1 → DATA STREAM
   ────────────────────────────
- Lit en continu depuis :
  hdfs:///project/raw/
- Obligatoire : readStream Spark Structured Streaming
- Contient images + labels

────────────────────────────
2. READ #2 → MODEL
   ────────────────────────────
- Charge un modèle ML (Spark MLlib ou modèle simulé)
- Ce modèle est utilisé pour la prédiction
- Peut être :
    - LogisticRegressionModel
    - RandomForestModel
    - ou scoring simulé (fallback accepté)

────────────────────────────
3. PROCESSING
   ────────────────────────────
   Sur chaque micro-batch :
- filter
- feature engineering
- prédiction ML :
  → ajout colonne "prediction"
  → ajout colonne "confidence_score"

────────────────────────────
4. WRITE #1 → PROCESSED DATA
   ────────────────────────────
   hdfs:///project/processed/
   Contient :
- données enrichies
- features
- labels
- metadata

────────────────────────────
5. WRITE #2 → PREDICTIONS OUTPUT
   ────────────────────────────
   Optionnel mais OBLIGATOIRE conceptuellement :
   hdfs:///project/processed/predictions/

Contient :
- image_path
- prediction
- confidence_score

────────────────────────────
6. STREAMING RULE
   ────────────────────────────
- append mode
- checkpoint obligatoire:
  hdfs:///project/checkpoint/
- awaitTermination()
- tourne indéfiniment

────────────────────────────────────────
HDFS STRUCTURE
────────────────────────────────────────
hdfs:///project/raw/
hdfs:///project/processed/
hdfs:///project/processed/predictions/
hdfs:///project/staging/
hdfs:///project/logs/
hdfs:///project/checkpoint/

────────────────────────────────────────
CONTRAINTES TECHNIQUES
────────────────────────────────────────
- Scala 2.12+
- Spark 3.x
- DataFrame API ONLY (NO RDD)
- Parquet obligatoire
- HDFS ONLY
- Producer = batch strict
- Consumer = streaming strict
- readStream obligatoire

────────────────────────────────────────
STREAMLIT
────────────────────────────────────────
- lit uniquement :
  hdfs:///project/processed/
  hdfs:///project/processed/predictions/

- aucun calcul ML
- uniquement visualisation :
    - dashboards
    - filtres
    - histogrammes de prédictions

────────────────────────────────────────
STRUCTURE PROJET
────────────────────────────────────────

/producer/src/main/scala/ProducerApp.scala
/consumer/src/main/scala/ConsumerApp.scala
/common/Utils.scala
/ml/ModelLoader.scala (optionnel)
/streamlit/app.py
/build.sbt

────────────────────────────────────────
OUTPUT ATTENDU
────────────────────────────────────────
1. Producer Scala complet (batch)
2. Consumer Scala complet (streaming + ML + double read + double write)
3. build.sbt complet
4. Streamlit app.py
5. explication du pipeline

────────────────────────────────────────
RÈGLES CRITIQUES
────────────────────────────────────────
- Producer exécute UNE SEULE FOIS
- Consumer tourne INFINIMENT
- Consumer lit :
  ✔ data (raw)
  ✔ model (ML)
- Consumer écrit :
  ✔ processed dataset
  ✔ predictions dataset
- readStream obligatoire
- architecture strictement respectée