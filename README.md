
# Heart Disease Classification MLOps Pipeline

A production-ready, configuration-driven Machine Learning operations (MLOps) pipeline that cleans raw cardiovascular patient metrics, versions the dataset layers, trains an optimized Random Forest Classifier, runs a 3-tier automated test suite, and serves real-time inferences.

## 🛠️ Tech Stack & Engineering Patterns

* **Core Framework:** Python 3.11+
* **Machine Learning & Monitoring:** Scikit-Learn, Pandas, NumPy, MLflow, Evidently AI
* **Automation & CI/CD:** GitHub Actions (Automated Test & Retrain quality gates)
* **Testing Framework:** Pytest (11 rigorous pipeline assertion checks)
* **Design Pattern:** Completely decoupled scripts driven by a centralized configuration layout (`configs/config.yaml`).

---

## 📂 Project Architecture

```text
heart disease/
├── .github/
│   └── workflows/
│       └── mlops.yml           # GitHub Actions 2-stage CI/CD configuration
├── configs/
│   └── config.yaml             # Centralized pipeline hyperparameters & paths
├── data/
│   ├── heart.csv               # Raw dataset
│   └── processed_heart.csv     # Cleaned, engineered dataset
├── models/
│   └── heart_classifier.pkl    # Serialized production model artifact
├── reports/
│   ├── metrics.json            # Evaluated pipeline validation metrics
│   └── data_drift_report.html  # Interactive Evidently AI drift dashboard
├── src/
│   ├── app.py                  # FastAPI web server and data validation schemas
│   ├── preprocess.py           # Data cleaning and missing value resolution
│   ├── train.py                # Model training and artifact extraction pipeline
│   ├── monitor_drift.py        # Evidently data drift threshold gate script
│   └── predict.py              # Local mock patient inference testing script
├── tests/
│   └── test_pipeline.py        # 3-tier Pytest suite (11/11 passing assertions)
└── MONITORING.md               # Written production drift analysis report
```
