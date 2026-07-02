import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_pipeline():
    print("--- Starting Data Preprocessing Stage ---")
    config = load_config()
    
    # 1. Load data dynamically from configuration paths
    raw_path = config["data"]["raw_path"]
    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset from {raw_path}. Shape: {df.shape}")
    
    # 2. Basic Cleaning / Handling Missing Values
    # Replacing common placeholder symbols if present, then dropping null rows
    df = df.replace("?", pd.NA).dropna()
    print(f"Dataset shape after removing missing values: {df.shape}")
    
    # 3. Create 'data' folder for output if it doesn't exist
    processed_path = config["data"]["processed_path"]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    # 4. Export clean dataset
    df.to_csv(processed_path, index=False)
    print(f"Successfully saved processed dataset to: {processed_path}")

if __name__ == "__main__":
    preprocess_pipeline()