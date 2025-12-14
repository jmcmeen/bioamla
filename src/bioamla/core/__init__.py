"""Core Module

This package contains the core functionality for bioamla including
audio processing, model inference, and data handling utilities.

Submodules:
    - ast: Audio Spectrogram Transformer model processing
    - datasets: Dataset management and validation
    - device: Device management for PyTorch operations
    - diagnostics: System diagnostics and information
    - fileutils: File utility functions
    - globals: Global constants
    - inat: iNaturalist audio data importing
    - inference: AST model inference
    - logging: Logging configuration
    - metadata: Metadata management
    - torchaudio: Audio processing utilities
    - training: AST model training

Note: Heavy dependencies (torch, transformers) are imported lazily in submodules.
Import from specific submodules as needed:
    from bioamla.core.metadata import read_metadata_csv
    from bioamla.core.device import get_device
"""
