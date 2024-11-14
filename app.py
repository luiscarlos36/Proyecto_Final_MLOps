from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado (debería haberse guardado previamente con joblib o MLflow)
model = joblib.load("best_rf_model.pkl")

app = FastAPI()

# Definir el esquema de datos de entrada
class HouseFeatures(BaseModel):
    # Definir las características de la casa (aquí deberías incluir todas las variables necesarias)
    overall_quality: int
    gr_liv_area: float
    garage_area: float
    # (añadir otras características...)

# Crear la ruta para recibir la predicción
@app.post("/predict/")
def predict(features: HouseFeatures):
    # Convertir las características de la casa en un array de numpy
    input_data = np.array([[features.overall_quality, features.gr_liv_area, features.garage_area]])
    # Realizar la predicción
    prediction = model.predict(input_data)
    return {"predicted_price": prediction[0]}
