# training_service/trainer.py
from fastapi import FastAPI
import torch
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import os
from typing import Dict

app = FastAPI()

class Trainer:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    async def train(self):
        """Train the model"""
        # Load data from MongoDB
        client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGO_URI"])
        db = client.prediction_service
        
        # Fetch training data
        cursor = db.predictions.find(
            {"actual_class": {"$exists": True}},
            batch_size=32
        )
        
        # Create data loaders
        train_loader = self._create_data_loader(cursor)
        
        # Train model
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self.model.to(self.device)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(10):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Save model
        await self._save_model()
    
    async def _save_model(self):
        """Save trained model to storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/model_{timestamp}.pth"
        
        torch.save(self.model.state_dict(), model_path)
        
        # Upload to storage
        async with httpx.AsyncClient() as client:
            with open(model_path, "rb") as f:
                await client.post(
                    os.environ["MODEL_STORAGE_URL"],
                    files={"model": f}
                )