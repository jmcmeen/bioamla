"""
Audio Classification Controller

This module contains controller functions for managing audio classification workflows.
It provides high-level orchestration of the audio classification pipeline, including
file validation, audio preprocessing, model inference, and response formatting.

The controller acts as an intermediary between the API layer and the core audio
processing functionality, handling error cases and ensuring proper data flow
through the classification pipeline.
"""

from pathlib import Path
from typing import Any, Optional, Union
from bioamla.core.exceptions import (
    NoModelLoadedError,
    UnsupportAudioFormatError
)
from bioamla.core.models.responses import AudioClassificationResponse
from bioamla.core.models.config import DefaultConfig
from bioamla.core.helpers.wav_torch import load_audio_from_bytes
from bioamla.core.helpers.transformers import process_audio_with_pipeline

async def classify_audio(
    model: Any, 
    audio_pipeline: Any, 
    file: Any, 
    top_k: int = 5
) -> AudioClassificationResponse:
    """
    Classify an uploaded audio file using the loaded model pipeline.
    
    This function orchestrates the complete audio classification workflow:
    1. Validates model and pipeline availability
    2. Checks file format compatibility
    3. Loads and preprocesses audio data
    4. Applies duration limits and trimming if necessary
    5. Runs classification inference
    6. Returns structured response with predictions and metadata
    
    Args:
        model (Any): Loaded audio classification model
        audio_pipeline (Any): Preprocessing pipeline for audio data
        file (Any): Uploaded file object with .filename and .read() method
        top_k (int): Number of top predictions to return (default: 5)
    
    Returns:
        AudioClassificationResponse: Structured response containing:
            - success: Boolean indicating operation success
            - predictions: List of top-k classification predictions
            - audio_duration: Duration of processed audio in seconds
            - sample_rate: Audio sample rate in Hz
            - model_used: Name of the model used for classification
            - processing_time: Total processing time in seconds
    
    Raises:
        NoModelLoadedError: If model or audio pipeline is not loaded
        UnsupportAudioFormatError: If uploaded file format is not supported
        Exception: For other processing errors during classification
    
    Note:
        Audio files longer than DefaultConfig.MAX_AUDIO_LENGTH will be
        automatically trimmed to fit within the maximum duration limit.
        
    Supported formats: .wav, .mp3, .flac, .ogg, .m4a
    """
    import time
    start_time = time.time()
    
    # Step 1: Validate that model and pipeline are loaded
    if model is None or audio_pipeline is None:
        raise NoModelLoadedError()
    
    # Step 2: Validate file type against supported formats
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise UnsupportAudioFormatError()
    
    try:
        # Step 3: Read audio file data
        audio_bytes = await file.read()
        
        # Step 4: Load and preprocess audio from bytes
        audio_array, sample_rate = load_audio_from_bytes(audio_bytes)
        
        # Step 5: Calculate audio duration
        duration = len(audio_array) / sample_rate
        
        # Step 6: Apply duration limits and trim if necessary
        if duration > DefaultConfig.MAX_AUDIO_LENGTH:
            # Trim audio to maximum allowed length
            max_samples = int(DefaultConfig.MAX_AUDIO_LENGTH * sample_rate)
            audio_array = audio_array[:max_samples]
            duration = DefaultConfig.MAX_AUDIO_LENGTH
        
        # Step 7: Run classification inference
        predictions = process_audio_with_pipeline(audio_array, sample_rate, top_k)
        
        # Step 8: Calculate total processing time
        processing_time = time.time() - start_time
        
        # Step 9: Return structured response
        return AudioClassificationResponse(
            success=True,
            predictions=predictions,
            audio_duration=duration,
            sample_rate=sample_rate,
            model_used=DefaultConfig.MODEL_NAME,
            processing_time=processing_time
        )
    except Exception:
        # Re-raise exceptions for proper error handling at higher levels
        # TODO: Add more specific exception handling for different error types
        raise
    
