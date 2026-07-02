import os
import yaml
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_dvc_hash(lock_path="dvc.lock"):
    """Programmatically extracts the raw data dependency hash from dvc.lock"""
    try:
        with open(lock_path, "r") as f:
            lock_data = yaml.safe_load(f)
            # Pull the MD5 hash of the raw data file dependency
            return lock_data["stages"]["preprocess"]["deps"][2]["md5"]
    except Exception:
        return "local_development_version"

def train_pipeline():
    print("--- Starting Model Training Stage with MLflow ---")
    config = load_config()
    
    # 1. Load Processed Data
    processed_path = config["data"]["processed_path"]
    df = pd.read_csv(processed_path)
    
    # 2. Split Features and Target
    target_col = config["data"]["target_col"]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    y = (y > 0).astype(int)
    
    # 3. Train/Test Split
    test_size = config["split"]["test_size"]
    rand_state = config["split"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rand_state
    )
    
    # 4. Set up MLflow Experiment Tracking
    mlflow.set_experiment("Heart_Disease_Classification")
    
    with mlflow.start_run():
        # A. Log All Hyperparameters from config.yaml
        mlflow.log_param("model_type", config["train"]["model_type"])
        mlflow.log_param("n_estimators", config["train"]["estimators"])
        mlflow.log_param("max_depth", config["train"]["max_depth"])
        mlflow.log_param("random_state", config["train"]["random_state"])
        
        # B. Log Data Version (DVC File Hash)
        data_version = get_dvc_hash()
        mlflow.log_param("dvc_data_version_hash", data_version)
        print(f"Logging data version hash to MLflow: {data_version}")
        
        # C. Initialize and Train Model
        print(f"Training a {config['train']['model_type']}...")
        model = RandomForestClassifier(
            n_estimators=config["train"]["estimators"],
            max_depth=config["train"]["max_depth"],
            random_state=config["train"]["random_state"]
        )
        model.fit(X_train, y_train)
        
        # D. Evaluate Performance Metrics
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        print(f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # E. Log All Metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # F. Save Local Model Artifact
        model_path = config["artifacts"]["model_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # G. Save Local Performance Reports
        metrics_path = config["artifacts"]["metrics_path"]
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        metrics = {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, predictions, output_dict=True)
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        # H. Log Model directly as an MLflow Artifact
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("Successfully logged hyperparameters, metrics, and model artifact to MLflow.")

if __name__ == "__main__":
    train_pipeline()