# 🚗 License Plate Detection — Pipeline Spark Batch + Streaming

Pipeline Big Data Scala/Spark conforme à `.claude/prompt.md` :
**Batch ingestion → HDFS RAW → Streaming processing → HDFS PROCESSED → Streamlit (visualisation)**.

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour le schéma d'archi et le flux de données détaillé.

---

## Architecture

```
plate-detection-spark/
└── 1-spark-preprocessing/
    ├── producer/   ← Batch (Spark Core) : images+labels YOLO -> letterbox -> raw/
    ├── consumer/   ← Streaming (Structured Streaming) : raw/ -> filtre/enrichit -> processed/
    ├── common/     ← Schémas, chemins HDFS, prétraitement image
    ├── streamlit/app.py ← Dashboard visu uniquement (lit processed/)
    └── build.sbt
```

---

## 🚀 Exécution

```bash
cd 1-spark-preprocessing
sbt clean assembly

# Producer (batch, une seule exécution) -> hdfs:///project/raw/
spark-submit --class ProducerApp --master yarn \
  producer/target/scala-2.12/producer-assembly-0.1.0.jar \
  hdfs:///project/staging/dataset

# Consumer (streaming, tourne en continu) -> hdfs:///project/processed/
spark-submit --class ConsumerApp --master yarn \
  consumer/target/scala-2.12/consumer-assembly-0.1.0.jar

# Dashboard (visu uniquement, lit hdfs:///project/processed/)
cd streamlit && pip install -r requirements.txt && streamlit run app.py
```

Voir `1-spark-preprocessing/README.md` pour le détail.

---

## 📊 Technologies

| Couche | Technologies |
|---|---|
| Producer (batch) | Apache Spark 3.x (Scala, Spark Core), Parquet, HDFS |
| Consumer (streaming) | Spark Structured Streaming, checkpointing, Parquet, HDFS |
| Visualisation | Streamlit (lecture seule de `hdfs:///project/processed/`) |
