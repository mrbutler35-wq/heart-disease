
# Heart Disease Classification MLOps Pipeline

A production-ready, configuration-driven Machine Learning operations (MLOps) pipeline that cleans raw cardiovascular patient metrics, versions the dataset layers, trains an optimized Random Forest Classifier, and serves real-time inferences via an asynchronous FastAPI microservice.

## 🛠️ Tech Stack & Engineering Patterns

* **Core Framework:** Python 3.13 / Anaconda
* **Machine Learning:** Scikit-Learn, Pandas, Joblib
* **Data Version Control (DVC):** Tracks data state changes, hashes artifacts, manages pipeline orchestration, and caches local storage backups.
* **API Layer:** FastAPI, Pydantic (data validation schema), Uvicorn (ASGI web server).
* **Design Pattern:** Completely decoupled scripts driven by a centralized configuration layout (`configs/config.yaml`).

---

## 📂 Project Architecture

```text
HE/
├── .dvc/               # Internal DVC tracking configurations
├── configs/
│   └── config.yaml     # Centralized pipeline hyperparameters & paths
├── data/
│   ├── heart.csv       # Raw dataset (Git ignored, DVC tracked)
│   ├── heart.csv.dvc   # DVC dataset tracking pointer
│   └── processed_heart.csv
├── models/
│   └── heart_classifier.pkl
├── reports/
│   └── metrics.json    # Evaluated pipeline validation metrics
└── src/
    ├── app.py          # FastAPI web server and data validation schemas
    ├── preprocess.py   # Data cleaning and missing value resolution
    ├── predict.py      # Local mock patient inference testing script
    └── train.py        # Model training and artifact extraction pipeline
```
