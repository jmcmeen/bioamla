"""
Audio Spectrogram Transformer Web API
======================================
A FastAPI-based web service for audio classification using 
Audio Spectrogram Transformer (AST) models from Hugging Face.
"""

import argparse
import uvicorn
import io
import base64
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile

import torch
import torchaudio
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    AutoFeatureExtractor,
    ASTForAudioClassification,
    pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Audio Spectrogram Transformer API",
    description="API for audio classification using AST models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and pipeline
model = None
feature_extractor = None
audio_pipeline = None
device = None

# Configuration
class Config:
    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 30  # seconds
    MIN_CONFIDENCE = 0.01
    TOP_K = 5

# Response models
class PredictionResult(BaseModel):
    label: str
    score: float
    rank: int

class AudioClassificationResponse(BaseModel):
    success: bool
    predictions: List[PredictionResult]
    audio_duration: Optional[float] = None
    sample_rate: int
    model_used: str
    processing_time: Optional[float] = None

class Base64AudioRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    top_k: Optional[int] = Field(default=5, description="Number of top predictions to return")

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None

# Helper functions
def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = Config.SAMPLE_RATE):
    """Load audio from bytes and resample if necessary."""
    try:
        # Create a file-like object from bytes
        audio_io = io.BytesIO(audio_bytes)
        
        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(audio_io)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sr
            )
            waveform = resampler(waveform)
        
        # Flatten to 1D array
        audio_array = waveform.squeeze().numpy()
        
        return audio_array, target_sr
    except Exception as e:
        logger.error(f"Error loading audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to load audio: {str(e)}")

def process_audio_with_pipeline(audio_array: np.ndarray, sample_rate: int, top_k: int = 5):
    """Process audio using the transformer pipeline."""
    try:
        # Run inference
        results = audio_pipeline(
            audio_array,
            sampling_rate=sample_rate,
            top_k=top_k
        )
        
        # Format results
        predictions = []
        for i, pred in enumerate(results):
            predictions.append(
                PredictionResult(
                    label=pred['label'],
                    score=float(pred['score']),
                    rank=i + 1
                )
            )
        
        return predictions
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

def process_audio_manual(audio_array: np.ndarray, sample_rate: int, top_k: int = 5):
    """Process audio manually using model and feature extractor."""
    try:
        # Extract features
        inputs = feature_extractor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], k=min(top_k, probs.shape[-1]))
        
        # Format results
        predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            label = model.config.id2label[idx.item()]
            predictions.append(
                PredictionResult(
                    label=label,
                    score=float(prob.item()),
                    rank=i + 1
                )
            )
        
        return predictions
    except Exception as e:
        logger.error(f"Error during manual inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Manual inference failed: {str(e)}")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global model, feature_extractor, audio_pipeline, device
    
    try:
        logger.info("Loading Audio Spectrogram Transformer model...")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(Config.MODEL_NAME)
        model = ASTForAudioClassification.from_pretrained(Config.MODEL_NAME)
        model.to(device)
        model.eval()
        
        # Create pipeline for easier inference
        audio_pipeline = pipeline(
            "audio-classification",
            model=Config.MODEL_NAME,
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Audio Spectrogram Transformer API",
        "version": "1.0.0",
        "model": Config.MODEL_NAME,
        "device": str(device),
        "endpoints": {
            "/classify": "POST - Classify audio file",
            "/classify-base64": "POST - Classify base64 encoded audio",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized"
    }

@app.get("/model-info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": Config.MODEL_NAME,
        "num_labels": model.config.num_labels,
        "sample_rate": Config.SAMPLE_RATE,
        "max_audio_length": Config.MAX_AUDIO_LENGTH,
        "device": str(device),
        "labels_sample": list(model.config.id2label.values())[:10]
    }

@app.post("/classify", response_model=AudioClassificationResponse)
async def classify_audio(
    file: UploadFile = File(..., description="Audio file to classify"),
    top_k: Optional[int] = Form(default=5, description="Number of top predictions")
):
    """Classify an uploaded audio file."""
    import time
    start_time = time.time()
    
    if model is None or audio_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read audio file
        audio_bytes = await file.read()
        
        # Load and preprocess audio
        audio_array, sample_rate = load_audio_from_bytes(audio_bytes)
        
        # Calculate duration
        duration = len(audio_array) / sample_rate
        
        # Check audio length
        if duration > Config.MAX_AUDIO_LENGTH:
            # Trim audio to max length
            max_samples = int(Config.MAX_AUDIO_LENGTH * sample_rate)
            audio_array = audio_array[:max_samples]
            duration = Config.MAX_AUDIO_LENGTH
        
        # Process audio
        predictions = process_audio_with_pipeline(audio_array, sample_rate, top_k)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return AudioClassificationResponse(
            success=True,
            predictions=predictions,
            audio_duration=duration,
            sample_rate=sample_rate,
            model_used=Config.MODEL_NAME,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/classify-base64", response_model=AudioClassificationResponse)
async def classify_audio_base64(request: Base64AudioRequest):
    """Classify base64 encoded audio."""
    import time
    start_time = time.time()
    
    if model is None or audio_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Load and preprocess audio
        audio_array, sample_rate = load_audio_from_bytes(audio_bytes)
        
        # Calculate duration
        duration = len(audio_array) / sample_rate
        
        # Check audio length
        if duration > Config.MAX_AUDIO_LENGTH:
            # Trim audio to max length
            max_samples = int(Config.MAX_AUDIO_LENGTH * sample_rate)
            audio_array = audio_array[:max_samples]
            duration = Config.MAX_AUDIO_LENGTH
        
        # Process audio
        predictions = process_audio_with_pipeline(audio_array, sample_rate, request.top_k)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return AudioClassificationResponse(
            success=True,
            predictions=predictions,
            audio_duration=duration,
            sample_rate=sample_rate,
            model_used=Config.MODEL_NAME,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing base64 audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/classify-batch")
async def classify_batch(
    files: List[UploadFile] = File(..., description="Multiple audio files to classify"),
    top_k: Optional[int] = Form(default=5, description="Number of top predictions per file")
):
    """Classify multiple audio files in batch."""
    if model is None or audio_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    errors = []
    
    for i, file in enumerate(files):
        try:
            # Read audio file
            audio_bytes = await file.read()
            
            # Load and preprocess audio
            audio_array, sample_rate = load_audio_from_bytes(audio_bytes)
            
            # Process audio
            predictions = process_audio_with_pipeline(audio_array, sample_rate, top_k)
            
            results.append({
                "filename": file.filename,
                "index": i,
                "predictions": [p.dict() for p in predictions]
            })
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "index": i,
                "error": str(e)
            })
    
    return {
        "success": len(errors) == 0,
        "total_files": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

def main():
    """Main function to run the server."""
    parser = argparse.ArgumentParser(description="Transformers Model API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--generate-client", action="store_true", help="Generate client example code")
    
    args = parser.parse_args()
    
    if args.generate_client:
        client_code = create_client_example()
        with open("client_example.py", "w") as f:
            f.write(client_code)
        print("Client example code generated: client_example.py")
        return
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Device: {device}")
    logger.info(f"Workers: {args.workers}")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()