"""
Custom Exception Classes for Audio Processing

This module defines custom exception classes used throughout the Bioamla audio
processing system. These exceptions provide specific error handling for common
audio processing and model loading scenarios.

The exceptions follow Python's standard exception hierarchy and include
meaningful default messages while allowing customization when needed.
"""

from typing import Optional

class UnsupportAudioFormatError(Exception):
    """
    Exception raised when an unsupported audio format is encountered.
    
    This exception is raised when attempting to process audio files in formats
    that are not supported by the current audio processing pipeline. Common
    scenarios include unsupported file extensions or corrupted audio files.
    
    Attributes:
        message (str): Explanation of the error
    """

    def __init__(self, message: str = "Audio format is not supported.") -> None:
        """
        Initialize the UnsupportAudioFormatError.
        
        Args:
            message (str): Custom error message (default: "Audio format is not supported.")
        """
        super().__init__(message)
        self.message = message

class NoModelLoadedError(Exception):
    """
    Exception raised when attempting to use a model that hasn't been loaded.
    
    This exception is raised when trying to perform model inference or other
    model-dependent operations without first loading the required model into
    memory. This typically occurs in API endpoints or processing functions
    that require a pre-loaded model.
    
    Attributes:
        message (str): Explanation of the error
    """

    def __init__(self, message: str = "No model is loaded.") -> None:
        """
        Initialize the NoModelLoadedError.
        
        Args:
            message (str): Custom error message (default: "No model is loaded.")
        """
        super().__init__(message)
        self.message = message

class AudioProcessingError(Exception):
    """
    Exception raised for general audio processing errors.
    
    This exception serves as a base class for audio processing related errors
    that don't fall into more specific categories. It can be used for
    transformation errors, filter failures, or other processing issues.
    
    Attributes:
        message (str): Explanation of the error
        operation (Optional[str]): The operation that failed
    """

    def __init__(
        self, 
        message: str = "An error occurred during audio processing.", 
        operation: Optional[str] = None
    ) -> None:
        """
        Initialize the AudioProcessingError.
        
        Args:
            message (str): Custom error message
            operation (Optional[str]): Name of the operation that failed
        """
        if operation:
            full_message = f"{message} Operation: {operation}"
        else:
            full_message = message
        
        super().__init__(full_message)
        self.message = message
        self.operation = operation

class ModelLoadError(Exception):
    """
    Exception raised when model loading fails.
    
    This exception is raised when there are issues loading machine learning
    models, including file not found errors, corrupted model files, or
    incompatible model formats.
    
    Attributes:
        message (str): Explanation of the error
        model_path (Optional[str]): Path to the model that failed to load
    """

    def __init__(
        self, 
        message: str = "Failed to load model.", 
        model_path: Optional[str] = None
    ) -> None:
        """
        Initialize the ModelLoadError.
        
        Args:
            message (str): Custom error message
            model_path (Optional[str]): Path to the model file
        """
        if model_path:
            full_message = f"{message} Model path: {model_path}"
        else:
            full_message = message
        
        super().__init__(full_message)
        self.message = message
        self.model_path = model_path