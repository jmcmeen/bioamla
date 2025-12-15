"""
BioAmla - Bioacoustics & Machine Learning Applications
======================================================

A comprehensive Python package for bioacoustic analysis and machine learning applications.
This package provides tools for audio processing, feature extraction, model training,
and inference specifically designed for wildlife sound analysis.

Key Features:
- Audio file processing and manipulation
- Audio Spectrogram Transformer (AST) model integration
- Batch processing capabilities
- Web API for audio classification
- Command-line tools for various audio tasks

Submodules:
    - ast: Audio Spectrogram Transformer model processing
    - augment: Audio data augmentation
    - datasets: Dataset management and validation
    - device: Device management for PyTorch operations
    - diagnostics: System diagnostics and information
    - evaluate: Model evaluation utilities
    - explore: Data exploration tools
    - fileutils: File utility functions
    - globals: Global constants
    - inat: iNaturalist audio data importing
    - inference: AST model inference
    - license: License management
    - logging: Logging configuration
    - metadata: Metadata management
    - signal: Audio signal processing
    - torchaudio: Audio processing utilities
    - training: AST model training
    - tui: Text User Interface
    - visualize: Audio visualization
    - wildlife_acoustics: Wildlife Acoustics integration

Note: Heavy dependencies (torch, transformers) are imported lazily in submodules.
Import from specific submodules as needed:
    from bioamla.metadata import read_metadata_csv
    from bioamla.device import get_device

Version: 0.0.49
"""
__version__ = "0.0.49"
