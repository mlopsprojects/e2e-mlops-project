from fastapi import FastAPI
import pickle
import os

app = FastAPI()

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app with the trained model"}

@app.post("/predict")
def predict(data: dict):
    # Assuming the model expects a dictionary input
    prediction = model.predict([data])
    return {"prediction": prediction}
