# backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import uvicorn
from typing import Optional
from pydantic import BaseModel
import motor.motor_asyncio
from config import settings

app = FastAPI()

# MongoDB connection
client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGO_URI)
db = client.prediction_service

class PredictionRequest(BaseModel):
    prediction: str
    confidence: float
    actual_class: Optional[str]

async def resize_image(image_data):
    """Resize image to 32x32 pixels"""
    image = Image.open(io.BytesIO(image_data))
    return image.resize((32, 32))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image data
    contents = await file.read()
    
    # Resize image
    resized_image = await resize_image(contents)
    
    # Call prediction service
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.PREDICTION_SERVICE_URL}/predict",
            content=resized_image.tobytes(),
            headers={"Content-Type": "application/octet-stream"}
        )
        
        if response.status_code == 200:
            prediction_data = response.json()
            
            # Store prediction in MongoDB
            await db.predictions.insert_one({
                "prediction": prediction_data["prediction"],
                "confidence": prediction_data["confidence"],
                "timestamp": datetime.utcnow(),
                "actual_class": None
            })
            
            return JSONResponse(prediction_data)
        else:
            return JSONResponse(
                {"error": "Prediction failed"},
                status_code=500
            )

@app.post("/report-error")
async def report_error(data: PredictionRequest):
    prediction_id = await db.predictions.find_one(
        {
            "prediction": data.prediction,
            "confidence": data.confidence,
            "actual_class": None
        },
        sort=[("_id", -1)]
    )
    
    if prediction_id:
        await db.predictions.update_one(
            {"_id": prediction_id["_id"]},
            {"$set": {"actual_class": data.actual_class}}
        )
        
        # Trigger model retraining
        async with httpx.AsyncClient() as client:
            await client.post(f"{settings.TRAINING_SERVICE_URL}/trigger-retrain")
            
        return JSONResponse({"message": "Error reported successfully"})
    else:
        return JSONResponse(
            {"error": "Prediction not found"},
            status_code=404
        )