"""
Configuration Settings
======================

This module defines default configuration values used throughout the
bioamla package for audio processing and model inference.

These settings provide consistent defaults while allowing for
customization in different use cases.
"""


class DefaultConfig:
    """
    Default configuration parameters for bioamla audio processing.

    Attributes:
        MODEL_NAME (str): Default Hugging Face model identifier for AST
        SAMPLE_RATE (int): Standard sample rate for audio processing (Hz)
        MAX_AUDIO_LENGTH (int): Maximum audio length for processing (seconds)
        MIN_CONFIDENCE (float): Minimum confidence threshold for predictions
        TOP_K (int): Default number of top predictions to return
    """

    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 30  # seconds
    MIN_CONFIDENCE = 0.01
    TOP_K = 5
