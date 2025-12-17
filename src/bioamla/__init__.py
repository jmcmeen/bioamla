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
- Custom classifier training (CNN, CRNN, Attention)
- Multi-label hierarchical classification
- Model ensemble predictions
- Embedding-based clustering (HDBSCAN, k-means, DBSCAN)
- UMAP/t-SNE dimensionality reduction
- Novelty detection for discovering unknown sounds
- Real-time audio recording and detection
- Real-time spectrogram streaming
- eBird API integration
- PostgreSQL database export

Submodules:
    - ast: Audio Spectrogram Transformer model processing
    - augment: Audio data augmentation
    - clustering: Embedding clustering, dimensionality reduction, novelty detection
    - datasets: Dataset management and validation
    - device: Device management for PyTorch operations
    - diagnostics: System diagnostics and information
    - evaluate: Model evaluation utilities
    - explore: Data exploration tools
    - fileutils: File utility functions
    - globals: Global constants
    - inat: iNaturalist audio data importing
    - inference: AST model inference
    - integrations: External API integrations (eBird, PostgreSQL)
    - license: License management
    - logging: Logging configuration
    - metadata: Metadata management
    - ml: Advanced ML (custom classifiers, hierarchical classification, ensembles)
    - realtime: Real-time audio recording and spectrogram streaming
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
    from bioamla.clustering import reduce_dimensions, AudioClusterer
    from bioamla.ml import Ensemble, HierarchicalClassifier
    from bioamla.realtime import LiveRecorder, RealtimeSpectrogram
    from bioamla.integrations import EBirdClient, PostgreSQLExporter

Version: 0.1.0
"""
__version__ = "0.1.0"
