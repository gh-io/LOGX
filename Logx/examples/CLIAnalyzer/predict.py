from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Request model
class FeaturesRequest(BaseModel):
    features: List[str]

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a Hugging Face pipeline (text classification example)
# You can replace "distilbert-base-uncased-finetuned-sst-2-english" with your own model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Prediction endpoint
@app.post("/predict")
async def get_prediction(request: FeaturesRequest):
    text_input = " ".join(request.features)  # combine features into a single string
    results = classifier(text_input)
    # Return the top prediction
    prediction = results[0]["label"]
    score = results[0]["score"]
    return {"prediction": f"{prediction} ({score:.2f})"}
