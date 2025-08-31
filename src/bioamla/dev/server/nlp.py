#!/usr/bin/env python3
"""
Transformers Endpoint Example Script
===================================

This script demonstrates how to create a production-ready API endpoint
for serving Hugging Face Transformers models using FastAPI.

Features:
- Multiple model endpoints (sentiment, NER, QA, summarization)
- Batch processing support
- Error handling and validation
- Performance monitoring
- GPU/CPU support
- Model caching and optimization

Usage:
    python transformers_endpoint.py --port 8000 --host 0.0.0.0
"""

import argparse
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from transformers import (
    pipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for models
models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pydantic models for request/response validation
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text to process")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to process")
    
    @validator('texts')
    def validate_texts(cls, v):
        processed_texts = []
        for text in v:
            if not text or not text.strip():
                raise ValueError('All texts must be non-empty')
            if len(text) > 10000:
                raise ValueError('Text length cannot exceed 10000 characters')
            processed_texts.append(text.strip())
        return processed_texts

class QuestionAnswerInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Question to ask")
    context: str = Field(..., min_length=1, max_length=5000, description="Context to search for answer")

class SentimentResponse(BaseModel):
    label: str
    score: float
    processing_time: float

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processing_time: float
    texts_processed: int

class EntityResponse(BaseModel):
    entity: str
    label: str
    confidence: float
    start: int
    end: int

class NERResponse(BaseModel):
    entities: List[EntityResponse]
    processing_time: float

class QAResponse(BaseModel):
    answer: str
    confidence: float
    start: int
    end: int
    processing_time: float

class SummarizationResponse(BaseModel):
    summary: str
    compression_ratio: float
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    device: str
    memory_usage: Dict[str, Any]

# Model management class
class ModelManager:
    """Manages loading and caching of models."""
    
    def __init__(self):
        self.models = {}
        self.device = device
        self.load_times = {}
    
    async def load_model(self, model_name: str, task: str, model_id: Optional[str] = None):
        """Load a model asynchronously."""
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return self.models[model_name]
        
        logger.info(f"Loading model {model_name} for task {task}...")
        start_time = time.time()
        
        try:
            if model_id:
                model = pipeline(task, model=model_id, device=0 if self.device == "cuda" else -1)
            else:
                model = pipeline(task, device=0 if self.device == "cuda" else -1)
            
            self.models[model_name] = model
            load_time = time.time() - start_time
            self.load_times[model_name] = load_time
            
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    def get_model(self, model_name: str):
        """Get a loaded model."""
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
        return self.models[model_name]
    
    def get_memory_usage(self):
        """Get current memory usage."""
        memory_info = {
            "device": self.device,
            "models_loaded": len(self.models),
            "model_names": list(self.models.keys())
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            })
        
        return memory_info

# Initialize model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Transformers API server...")
    logger.info(f"Using device: {device}")
    
    # Load default models
    await model_manager.load_model("sentiment", "sentiment-analysis")
    await model_manager.load_model("ner", "ner")
    await model_manager.load_model("qa", "question-answering")
    await model_manager.load_model("summarization", "summarization")
    
    logger.info("All models loaded successfully!")
    yield
    
    logger.info("Shutting down...")
    # Cleanup if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Create FastAPI app
app = FastAPI(
    title="Transformers Model API",
    description="Production-ready API for Hugging Face Transformers models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def measure_time(func):
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
        return result, processing_time
    return wrapper

@measure_time
def run_sentiment_analysis(model, text: str):
    """Run sentiment analysis with timing."""
    return model(text)[0]

@measure_time
def run_ner(model, text: str):
    """Run NER with timing."""
    return model(text)

@measure_time
def run_qa(model, question: str, context: str):
    """Run question answering with timing."""
    return model(question=question, context=context)

@measure_time
def run_summarization(model, text: str):
    """Run summarization with timing."""
    max_length = min(150, len(text.split()) // 2)
    min_length = max(30, max_length // 3)
    return model(text, max_length=max_length, min_length=min_length, do_sample=False)[0]

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Transformers Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        memory_usage = model_manager.get_memory_usage()
        return HealthResponse(
            status="healthy",
            models_loaded=list(model_manager.models.keys()),
            device=device,
            memory_usage=memory_usage
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """Analyze sentiment of input text."""
    try:
        model = model_manager.get_model("sentiment")
        result, processing_time = run_sentiment_analysis(model, input_data.text)
        
        return SentimentResponse(
            label=result["label"],
            score=result["score"],
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.post("/sentiment/batch", response_model=BatchSentimentResponse)
async def analyze_sentiment_batch(input_data: BatchTextInput):
    """Analyze sentiment of multiple texts."""
    try:
        model = model_manager.get_model("sentiment")
        start_time = time.time()
        
        # Process in batch for efficiency
        batch_results = model(input_data.texts)
        
        total_time = time.time() - start_time
        
        results = [
            SentimentResponse(
                label=result["label"],
                score=result["score"],
                processing_time=total_time / len(input_data.texts)  # Average time per text
            )
            for result in batch_results
        ]
        
        return BatchSentimentResponse(
            results=results,
            total_processing_time=total_time,
            texts_processed=len(input_data.texts)
        )
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch sentiment analysis failed: {str(e)}")

@app.post("/ner", response_model=NERResponse)
async def extract_entities(input_data: TextInput):
    """Extract named entities from text."""
    try:
        model = model_manager.get_model("ner")
        entities, processing_time = run_ner(model, input_data.text)
        
        entity_responses = [
            EntityResponse(
                entity=entity["word"],
                label=entity["entity_group"] if "entity_group" in entity else entity["entity"],
                confidence=entity["score"],
                start=entity["start"],
                end=entity["end"]
            )
            for entity in entities
        ]
        
        return NERResponse(
            entities=entity_responses,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"NER failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"NER failed: {str(e)}")

@app.post("/qa", response_model=QAResponse)
async def answer_question(input_data: QuestionAnswerInput):
    """Answer a question based on provided context."""
    try:
        model = model_manager.get_model("qa")
        result, processing_time = run_qa(model, input_data.question, input_data.context)
        
        return QAResponse(
            answer=result["answer"],
            confidence=result["score"],
            start=result["start"],
            end=result["end"],
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"QA failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(input_data: TextInput):
    """Summarize input text."""
    try:
        model = model_manager.get_model("summarization")
        result, processing_time = run_summarization(model, input_data.text)
        
        original_length = len(input_data.text)
        summary_length = len(result["summary_text"])
        compression_ratio = summary_length / original_length
        
        return SummarizationResponse(
            summary=result["summary_text"],
            compression_ratio=compression_ratio,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/models/load")
async def load_custom_model(
    model_name: str,
    task: str,
    model_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """Load a custom model."""
    try:
        if background_tasks:
            background_tasks.add_task(
                model_manager.load_model, 
                model_name, 
                task, 
                model_id
            )
            return {"message": f"Loading model {model_name} in background"}
        else:
            await model_manager.load_model(model_name, task, model_id)
            return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/models")
async def list_models():
    """List all loaded models."""
    return {
        "models": list(model_manager.models.keys()),
        "load_times": model_manager.load_times,
        "device": device
    }

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a specific model."""
    try:
        if model_name not in model_manager.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        del model_manager.models[model_name]
        if model_name in model_manager.load_times:
            del model_manager.load_times[model_name]
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"message": f"Model {model_name} unloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# Client example
def create_client_example():
    """Create example client code."""
    client_code = '''
# Example client code
import requests
import json

BASE_URL = "http://localhost:8000"

def test_sentiment(text):
    response = requests.post(
        f"{BASE_URL}/sentiment",
        json={"text": text}
    )
    return response.json()

def test_ner(text):
    response = requests.post(
        f"{BASE_URL}/ner",
        json={"text": text}
    )
    return response.json()

def test_qa(question, context):
    response = requests.post(
        f"{BASE_URL}/qa",
        json={"question": question, "context": context}
    )
    return response.json()

def test_batch_sentiment(texts):
    response = requests.post(
        f"{BASE_URL}/sentiment/batch",
        json={"texts": texts}
    )
    return response.json()

# Example usage
if __name__ == "__main__":
    # Test sentiment analysis
    result = test_sentiment("I love this API!")
    print("Sentiment:", result)
    
    # Test NER
    result = test_ner("Apple Inc. was founded by Steve Jobs.")
    print("NER:", result)
    
    # Test QA
    result = test_qa(
        "Who founded Apple?", 
        "Apple Inc. was founded by Steve Jobs in 1976."
    )
    print("QA:", result)
    
    # Test batch processing
    result = test_batch_sentiment([
        "Great product!",
        "Terrible service.",
        "It's okay."
    ])
    print("Batch sentiment:", result)
'''
    return client_code

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
