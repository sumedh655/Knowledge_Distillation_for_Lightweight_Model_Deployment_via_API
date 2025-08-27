from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import json
import os


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
MODEL_JSON = os.path.join(MODEL_DIR, "student_model.json")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "student_model.bin")


with open(MODEL_JSON, "r") as json_file:
    model_json = json_file.read()
student_model = tf.keras.models.model_from_json(model_json)


student_model.load_weights(MODEL_WEIGHTS)


student_model.trainable = False


app = FastAPI(title="Knowledge Distillation Student Model API")


class PredictionRequest(BaseModel):
    data: list  # Example: [[0.1, 0.2, 0.3, ...]]


@app.get("/")
def home():
    return {"message": "Student Model API is running!"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
      
        input_data = np.array(request.data, dtype=np.float32)

       
        predictions = student_model.predict(input_data).tolist()

        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}
