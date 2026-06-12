# Pipeline Spark Batch + Streaming (HDFS)

## Structure
```
4-spark-batch-streaming/
├── build.sbt
├── common/src/main/scala/Utils.scala      # paths HDFS, schema, scoring
├── producer/src/main/scala/ProducerApp.scala   # batch -> raw/
└── consumer/src/main/scala/ConsumerApp.scala   # streaming raw/ -> processed/
```

## HDFS
```
hdfs:///project/raw/         # sortie producer (parquet)
hdfs:///project/processed/   # sortie consumer (parquet, append)
hdfs:///project/staging/
hdfs:///project/logs/
hdfs:///project/checkpoint/  # checkpoints streaming
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
  producer/target/scala-2.12/producer-assembly-0.1.0.jar
```
Optionnel: passer un CSV en argument pour lire un dataset réel au lieu des données simulées :
```bash
spark-submit --class ProducerApp --master yarn jar.jar /path/dataset.csv
```
-> écrit dans `hdfs:///project/raw/` puis s'arrête (`spark.stop()`).

### Consumer (streaming, tourne en continu)
```bash
spark-submit --class ConsumerApp --master yarn \
  consumer/target/scala-2.12/consumer-assembly-0.1.0.jar
```
-> `readStream` sur `hdfs:///project/raw/`, filtre `confidence >= 0.5`, ajoute `risk_level` (HIGH/MEDIUM/LOW) + `processed_at`, écrit en append dans `hdfs:///project/processed/` avec checkpoint `hdfs:///project/checkpoint/processed/`.
Une 2ème requête streaming agrège par fenêtre 1min/vehicle_type vers `hdfs:///project/processed/aggregates/`.

## Flux de données
1. **Producer** génère/lit des détections de plaques, nettoie (dédup, plaques vides, confidence hors [0,1]), écrit une fois en Parquet dans `raw/`, s'arrête.
2. **Consumer** détecte les nouveaux fichiers Parquet dans `raw/` (micro-batch toutes les 10s), filtre + enrichit (risk_level), écrit en append dans `processed/`, et maintient en parallèle des agrégats fenêtrés dans `processed/aggregates/`. Tourne indéfiniment via `awaitAnyTermination()`.
