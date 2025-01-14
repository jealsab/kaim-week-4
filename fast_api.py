from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum
import numpy as np
import joblib  # Import joblib to load the saved model

# Load the trained LSTM model using joblib
model = joblib.load("lstm.pkl")  # Ensure this is the correct path to your model file

# Initialize the FastAPI app
app = FastAPI()
handler = Mangum(app)

# Define the request body structure
class PredictionRequest(BaseModel):
    features: list  # List of input features for prediction

# Define the home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the LSTM Model Prediction API"}

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        # Convert input features to NumPy array and reshape for LSTM
        input_data = np.array(data.features).reshape((1, len(data.features), 1))

        # Make prediction
        prediction = model.predict(input_data)

        # Return the prediction
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
