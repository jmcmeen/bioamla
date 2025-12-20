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

Package Structure:
    - audio: Audio processing (analysis, signal processing, torchaudio utilities)
    - analysis: Acoustic analysis (indices, clustering, exploration)
    - detection: Detection algorithms (AST inference, energy detectors, RIBBIT)
    - files: File operations (I/O, paths, downloads, discovery)
    - models: Model inference and evaluation
    - ml: Machine learning models (AST, BirdNET, OpenSoundscape wrappers)
    - services: External service integrations (iNaturalist, Xeno-canto, Macaulay, eBird)
    - core: Legacy core functionality (config, visualization, etc.)

Note: Heavy dependencies (torch, transformers) are imported lazily in submodules.
Import from specific submodules as needed:
    from bioamla.audio import analyze_audio, AudioInfo
    from bioamla.detection import BandLimitedEnergyDetector, RibbitDetector
    from bioamla.detection.ast import wav_ast_inference, load_pretrained_ast_model
    from bioamla.files import TextFile, get_files_by_extension
    from bioamla.ml import load_model, ASTModel
    from bioamla.services import search_inaturalist, download_xeno_canto
    from bioamla.utils import get_audio_files, SUPPORTED_AUDIO_EXTENSIONS

Version: 0.1.1
"""

__version__ = "0.1.1"

# Re-export commonly used utilities for convenience
from bioamla.core.utils import (
    SUPPORTED_AUDIO_EXTENSIONS,
    create_directory,
    directory_exists,
    file_exists,
    get_audio_files,
    get_files_by_extension,
)

__all__ = [
    "__version__",
    # Utilities
    "SUPPORTED_AUDIO_EXTENSIONS",
    "get_audio_files",
    "get_files_by_extension",
    "file_exists",
    "directory_exists",
    "create_directory",
]
