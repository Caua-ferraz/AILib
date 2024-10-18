from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import joblib
import uvicorn
from .error_handling import AILibError

app = FastAPI()

class PredictionRequest(BaseModel):
    data: Any

class PredictionResponse(BaseModel):
    prediction: Any

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model = joblib.load("path_to_saved_model.pkl")
        prediction = model.predict(request.data)
        return PredictionResponse(prediction=prediction.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)
