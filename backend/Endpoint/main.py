from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# Define request model
class FeaturesRequest(BaseModel):
    features: List[str]

# Initialize app
app = FastAPI()

# Enable CORS (so your frontend can talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Example predict function
def predict(features: List[str]) -> str:
    # Replace this with your actual AI logic
    return f"Predicted output for {len(features)} features"

# POST endpoint
@app.post("/predict")
async def get_prediction(request: FeaturesRequest):
    prediction = predict(request.features)
    return {"prediction": prediction}
