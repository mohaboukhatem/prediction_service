# prediction_service/predictor.py
from fastapi import FastAPI
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from typing import Dict

app = FastAPI()

class Predictor:
    def __init__(self):
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    async def load_model(self):
        """Load the latest model from storage"""
        # Implement model loading logic here
        pass
    
    async def predict(self, image_data: bytes) -> Dict:
        """Make prediction on input image"""
        if not self.model:
            await self.load_model()
            
        # Load and transform image
        image = Image.open(io.BytesIO(image_data))
        image = self.transform(image)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            
        return {
            "prediction": str(predicted.item()),
            "confidence": float(torch.max(torch.nn.functional.softmax(output, dim=1)).item())
        }

predictor = Predictor()

@app.post("/predict")
async def predict_endpoint(image_data: bytes):
    return await predictor.predict(image_data)