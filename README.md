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
│   └── build.sbt
│
├── 2-model-training/            ← Microservice 2 : Entraînement YOLOv8
│   ├── training.ipynb           ← Notebook Jupyter complet
│   ├── requirements.txt
│   └── models/                  ← Modèles entraînés & graphiques
│       ├── data.yaml
│       ├── best.pt              ← Meilleur modèle après entraînement
│       └── license_plate_yolov8n/
│
└── 3-web-interface/             ← Microservice 3 : Interface Web Flask
    ├── app.py                   ← API Flask (EDA + Prediction)
    ├── requirements.txt
    ├── model/
    │   └── best.pt              ← Modèle copié depuis MS2
    ├── templates/
    │   ├── index.html           ← Dashboard EDA
    │   └── predict.html         ← Page prédiction
    ├── static/
    │   ├── css/style.css
    │   └── js/main.js & predict.js
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
```
Voir `1-spark-preprocessing/README.md` pour le détail.

### Étape 2 — Entraînement du modèle
```bash
cd 2-model-training
pip install -r requirements.txt
jupyter notebook training.ipynb
# → Entraîne YOLOv8, exporte best.pt vers 3-web-interface/model/
```

### Étape 3 — Interface Web

- Pour lancer l'API :
```bash
cd 3-web-interface
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

- Pour lancer l'interface : 
```bash
cd 3-web-interface
cd StreamLit_App
streamlit run app.py
# → http://localhost:8502
```
---

## 🔗 Pages de l'interface web

| Route | Description |
|---|---|
| `http://localhost:5000/` | Dashboard EDA avec graphiques |
| `http://localhost:5000/predict` | Prédiction sur image uploadée |
| `http://localhost:5000/api/stats` | API JSON des statistiques |

---

## 📊 Technologies

| Couche | Technologies |
|---|---|
| Preprocessing | Apache Spark 3.x (Scala), Structured Streaming, HDFS, Parquet |
| Entraînement | YOLOv8 (Ultralytics), Pandas, Matplotlib |
| Interface | Flask, HTML/CSS/JS (Vanilla) |
