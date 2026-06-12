# 🚗 License Plate Detection — Projet Spark Core

Architecture **microservices** pour la détection de plaques d'immatriculation.

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour le schéma d'archi et le flux de données détaillé.

---

## Architecture

```
Projet Spark Core/
│
├── license-plate-detection-dataset-10125-images/   ← Dataset YOLO (train/valid/test)
│
├── 1-spark-preprocessing/      ← Microservice 1 : Prétraitement Spark (Scala)
│   ├── producer/   ← Batch : lecture images+labels YOLO, letterbox resize,
│   │                  features bbox, écriture unique -> hdfs:///project/raw/
│   ├── consumer/   ← Streaming : readStream raw/ -> filtre/enrichit -> processed/
│   ├── common/     ← Schémas, chemins HDFS, prétraitement image
│   ├── streamlit/app.py ← Dashboard visu uniquement (lit processed/)
│   └── build.sbt
│
├── 2-model-training/             ← Microservice 2 : Entraînement (PyTorch)
│   ├── train.py, dataloader.py, model.py, loss.py
│   ├── requirements.txt
│   └── checkpoints/
│       └── best_model.pt        ← Meilleur modèle après entraînement
│
└── 3-web-interface/              ← Microservice 3 : FastAPI + Streamlit
    ├── api.py                   ← Backend FastAPI (port 8000)
    ├── app.py                   ← Frontend Streamlit (port 8501)
    ├── inference.py             ← Charge ../2-model-training/checkpoints/best_model.pt
    ├── spark_batch.py           ← Inférence batch (PySpark, >3 images)
    ├── run.sh                   ← Lance api+streamlit (api|streamlit|both)
    ├── requirements.txt
    ├── history.csv              ← Historique des prédictions
    ├── images/                  ← original.jpg + annotated.jpg par prédiction
    └── uploads/                 ← Images uploadées temporairement
```

---

## 🚀 Ordre d'exécution

### Étape 1 — Prétraitement Spark (Scala, batch + streaming)
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

### Étape 2 — Entraînement du modèle
```bash
cd 2-model-training
pip install -r requirements.txt
python train.py
# → Entraîne le modèle, sauvegarde checkpoints/best_model.pt
```

### Étape 3 — Interface Web (FastAPI + Streamlit)
```bash
cd 3-web-interface
pip install -r requirements.txt

./run.sh           # FastAPI (8000) + Streamlit (8501)
# ou séparément :
./run.sh api        # FastAPI seule  -> http://localhost:8000 (docs: /docs)
./run.sh streamlit  # Streamlit seule -> http://localhost:8501
```
`inference.py` charge le modèle depuis `../2-model-training/checkpoints/best_model.pt`.

---

## 🔗 Endpoints de l'API (FastAPI, port 8000)

| Route | Description |
|---|---|
| `GET /` | Infos API |
| `GET /health`, `/api/health` | Health check |
| `GET /api/stats` | Statistiques JSON |
| `POST /predict`, `/api/predict` | Prédiction sur une image |
| `POST /predict_batch` | Prédiction batch (PySpark, `spark_batch.py`) |
| `GET /history` | Historique des prédictions |

L'UI Streamlit (port 8501) consomme ces endpoints.

---

## 📊 Technologies

| Couche | Technologies |
|---|---|
| Preprocessing | Apache Spark 3.x (Scala), Structured Streaming, HDFS, Parquet |
| Entraînement | PyTorch, Pandas, Matplotlib |
| Interface | FastAPI, Streamlit, PySpark (inférence batch) |
