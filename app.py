from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
from model import preprocess_image
import torch

from CNN_model import FirstCNN
from model import predict_image
from schemas import PredictionResponse, PredictionRequest

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FirstCNN()
model.load_state_dict(torch.load("CNN_model.pth", map_location=torch.device('cpu')))
model.eval()
possible_classes = ["cat", "dog"]  # Example


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image.")
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    pred_class = await predict_image(image_tensor)

    return PredictionResponse(
        prediction=pred_class,
        possible_classes=possible_classes
    )
