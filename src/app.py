import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Heart Disease Classification API",
    description="A production-ready MLOps API utilizing a Random Forest model to predict cardiovascular risk.",
    version="1.0.0"
)

# 2. Load Pipeline Settings from centralized YAML config
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
model_path = config["artifacts"]["model_path"]

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load the model binary at {model_path}. Error: {str(e)}")


# 3. Data Validation Layer: Structure of incoming JSON requests
class PatientData(BaseModel):
    age: float = Field(..., example=57.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=3.0)
    trestbps: float = Field(..., example=145.0)
    chol: float = Field(..., example=233.0)
    fbs: float = Field(..., example=1.0)
    restecg: float = Field(..., example=2.0)
    thalach: float = Field(..., example=150.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=2.3)
    slope: float = Field(..., example=3.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=6.0)


# 4. Documentation Layer: Structure of outgoing JSON responses
class PredictionResponse(BaseModel):
    prediction: int
    status: str
    confidence: float
    probabilities: dict


# 5. Root Health-Check Endpoint
@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": True, "version": "1.0.0"}


# 6. Production Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_heart_disease(patient: PatientData):
    try:
        # Convert incoming JSON object directly into a Pandas DataFrame shape
        patient_dict = {k: [v] for k, v in patient.dict().items()}
        input_df = pd.DataFrame(patient_dict)
        
        # Run inference against serialized Random Forest binary
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction])
        
        status_message = (
            "Clinical signs of HEART DISEASE detected." 
            if prediction == 1 
            else "HEALTHY - No immediate risk flags detected."
        )
        
        return {
            "prediction": prediction,
            "status": status_message,
            "confidence": confidence,
            "probabilities": {
                "healthy": float(probabilities[0]),
                "heart_disease": float(probabilities[1])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")