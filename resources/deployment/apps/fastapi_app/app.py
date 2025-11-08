"""
FastAPI Application for Model Serving
Lecture 83 - Deployment

This application serves a Fashion-MNIST CNN model via REST API.

Endpoints:
- GET /ping - Health check
- POST /predict - Make predictions
- GET /metadata - Model information
- GET /docs - Interactive API documentation

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fashion-MNIST CNN API",
    description="REST API for Fashion-MNIST image classification",
    version="1.0.0"
)

# Global variables for model and config
model = None
preprocessing_config = None
model_metadata = {}


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    instances: List[List[List[int]]] = Field(
        ...,
        description="List of 28x28 images (grayscale)",
        min_items=1,
        max_items=32
    )
    
    @validator('instances')
    def validate_shape(cls, v):
        """Validate image dimensions."""
        for img in v:
            if len(img) != 28 or any(len(row) != 28 for row in img):
                raise ValueError("Each image must be 28x28 pixels")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[dict]
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


@app.on_event("startup")
async def load_model():
    """Load model and preprocessing config on startup."""
    global model, preprocessing_config, model_metadata
    
    try:
        # Get model path from environment or use default
        model_dir = Path(os.getenv("MODEL_PATH", "../../models"))
        
        # Find the latest model (try .keras format first, then SavedModel directories, then .h5)
        keras_files = list(model_dir.glob("fashion_mnist_cnn_*.keras"))
        savedmodel_dirs = list(model_dir.glob("fashion_mnist_cnn_savedmodel_*"))
        h5_files = list(model_dir.glob("fashion_mnist_cnn_*.h5"))
        
        if keras_files:
            latest_model = sorted(keras_files)[-1]
            logger.info(f"Loading Keras model from: {latest_model}")
        elif savedmodel_dirs:
            latest_model = sorted(savedmodel_dirs)[-1]
            logger.info(f"Loading SavedModel from: {latest_model}")
        elif h5_files:
            latest_model = sorted(h5_files)[-1]
            logger.info(f"Loading H5 model from: {latest_model}")
        else:
            logger.error(f"No model found in {model_dir}")
            return
        
        model = tf.keras.models.load_model(latest_model)
        
        # Load preprocessing config
        config_files = list(model_dir.glob("preprocessing_config_*.pkl"))
        if config_files:
            latest_config = sorted(config_files)[-1]
            logger.info(f"Loading config from: {latest_config}")
            preprocessing_config = joblib.load(latest_config)
        
        # Load metadata if available
        metadata_files = list(model_dir.glob("model_metadata_*.json"))
        if metadata_files:
            import json
            latest_metadata = sorted(metadata_files)[-1]
            with open(latest_metadata, 'r') as f:
                model_metadata = json.load(f)
        
        logger.info("✓ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fashion-MNIST CNN API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/ping",
            "predict": "/predict",
            "metadata": "/metadata",
            "docs": "/docs"
        }
    }


@app.get("/ping", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.get("/metadata", response_model=dict)
async def get_metadata():
    """Get model metadata."""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_name": model_metadata.get("model_name", "fashion_mnist_cnn"),
        "version": model_metadata.get("version", "unknown"),
        "input_shape": preprocessing_config.get("input_shape", [28, 28, 1]) if preprocessing_config else [28, 28, 1],
        "output_classes": preprocessing_config.get("num_classes", 10) if preprocessing_config else 10,
        "class_names": preprocessing_config.get("class_names", []) if preprocessing_config else [],
        "framework": "tensorflow",
        "framework_version": tf.__version__
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions on input images.
    
    Args:
        request: PredictionRequest containing list of 28x28 images
        
    Returns:
        PredictionResponse with predictions and metadata
    """
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert to numpy array
        images = np.array(request.instances, dtype=np.float32)
        
        # Normalize
        images = images / 255.0
        
        # Reshape to (batch, 28, 28, 1)
        images = images.reshape(-1, 28, 28, 1)
        
        # Make predictions
        predictions = model.predict(images, verbose=0)
        
        # Format results
        class_names = preprocessing_config.get("class_names", []) if preprocessing_config else []
        
        results = []
        for pred in predictions:
            class_id = int(np.argmax(pred))
            confidence = float(pred[class_id])
            
            result = {
                "class_id": class_id,
                "class_name": class_names[class_id] if class_names else f"Class {class_id}",
                "confidence": confidence,
                "probabilities": pred.tolist()
            }
            results.append(result)
        
        return {
            "predictions": results,
            "model_version": model_metadata.get("version", "unknown")
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
