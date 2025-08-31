
"""
Audio Transformer Pipeline Helper Module

This module provides helper functions for processing audio data through machine learning
transformer pipelines. It handles the integration between raw audio data and pre-trained
models, including input preprocessing, inference execution, and output formatting.

The module is designed to work with Hugging Face transformers and provides a consistent
interface for audio classification tasks.
"""

import numpy as np
from typing import List, Any, Union
from bioamla.core.models.responses import PredictionResult

def process_audio_with_pipeline(
    audio_array: np.ndarray, 
    sample_rate: int, 
    top_k: int = 5
) -> List[PredictionResult]:
    """
    Process audio data through a transformer pipeline for classification.
    
    This function takes raw audio data and processes it through a pre-trained
    transformer model pipeline to generate classification predictions. It handles
    the interface between numpy audio arrays and the transformer pipeline,
    and formats the results into structured prediction objects.
    
    Args:
        audio_array (np.ndarray): Raw audio samples as a numpy array
        sample_rate (int): Sample rate of the audio in Hz
        top_k (int): Number of top predictions to return (default: 5)
    
    Returns:
        List[PredictionResult]: List of prediction results ordered by confidence score,
                               each containing label, score, and rank information
    
    Raises:
        Exception: If pipeline processing fails or audio_pipeline is not available
        ValueError: If audio_array is empty or invalid
        TypeError: If input parameters are not of expected types
    
    Example:
        >>> import numpy as np
        >>> audio_data = np.random.randn(16000)  # 1 second of audio at 16kHz
        >>> predictions = process_audio_with_pipeline(audio_data, 16000, top_k=3)
        >>> for pred in predictions:
        ...     print(f"{pred.label}: {pred.score:.3f}")
    
    Note:
        This function expects a global audio_pipeline object to be available.
        The pipeline should be a Hugging Face transformers pipeline configured
        for audio classification tasks.
    """
    # TODO: Add proper pipeline injection or global pipeline management
    # For now, this assumes audio_pipeline is available in the calling context
    from bioamla.core.models.config import DefaultConfig
    
    # This is a placeholder - the actual pipeline should be injected or managed globally
    # audio_pipeline = get_global_pipeline() or similar
    
    try:
        # Run inference through the transformer pipeline
        # This expects the audio_pipeline to be available in the calling scope
        results = audio_pipeline(
            audio_array,
            sampling_rate=sample_rate,
            top_k=top_k
        )
        
        # Format results into structured prediction objects
        predictions = []
        for i, pred in enumerate(results):
            # Create PredictionResult object for each prediction
            prediction = PredictionResult(
                label=pred['label'],
                score=float(pred['score']),
                rank=i + 1
            )
            predictions.append(prediction)
        
        return predictions
        
    except Exception:
        # TODO: Add more specific exception handling for different error types
        # Possible exceptions: model loading errors, audio format issues, etc.
        raise

def validate_audio_input(
    audio_array: np.ndarray, 
    sample_rate: int
) -> bool:
    """
    Validate audio input parameters for processing.
    
    Args:
        audio_array (np.ndarray): Audio samples array
        sample_rate (int): Sample rate in Hz
    
    Returns:
        bool: True if input is valid
    
    Raises:
        ValueError: If input validation fails
    """
    if not isinstance(audio_array, np.ndarray):
        raise ValueError("audio_array must be a numpy ndarray")
    
    if audio_array.size == 0:
        raise ValueError("audio_array cannot be empty")
    
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    
    return True

def format_pipeline_results(
    raw_results: List[dict], 
    include_rank: bool = True
) -> List[PredictionResult]:
    """
    Format raw pipeline results into structured prediction objects.
    
    Args:
        raw_results (List[dict]): Raw results from transformer pipeline
        include_rank (bool): Whether to include ranking information
    
    Returns:
        List[PredictionResult]: Formatted prediction results
    """
    formatted_results = []
    
    for i, result in enumerate(raw_results):
        prediction = PredictionResult(
            label=result['label'],
            score=float(result['score']),
            rank=i + 1 if include_rank else None
        )
        formatted_results.append(prediction)
    
    return formatted_results