# backend/config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    PREDICTION_SERVICE_URL: str
    TRAINING_SERVICE_URL: str
    MODEL_STORAGE_URL: str
    
    class Config:
        env_file = ".env"

settings = Settings()