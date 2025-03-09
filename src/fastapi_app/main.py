from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from FastAPI v1.0.4 !"}

@app.get("/data")
def get_data():
    return {"data": [1, 2, 3, 4, 5]}
