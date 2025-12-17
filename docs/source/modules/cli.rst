CLI Module
==========

Command-line interface for bioamla.

Command Structure
-----------------

The bioamla CLI is organized into the following command groups:

**Top-level commands:**

- ``bioamla version`` - Display version information
- ``bioamla devices`` - Display CUDA/GPU device information
- ``bioamla explore`` - Launch interactive TUI dashboard

**config** - Configuration management:

- ``bioamla config show`` - Show current configuration
- ``bioamla config init`` - Create default configuration file
- ``bioamla config path`` - Show configuration file search paths
- ``bioamla config purge`` - Purge cached HuggingFace Hub data

**audio** - Audio file utilities:

- ``bioamla audio info`` - Display audio file metadata
- ``bioamla audio convert`` - Convert audio formats
- ``bioamla audio filter`` - Apply frequency filters
- ``bioamla audio normalize`` - Normalize audio loudness
- ``bioamla audio resample`` - Resample to different sample rate
- ``bioamla audio trim`` - Trim audio files
- ``bioamla audio segment`` - Split audio on silence
- ``bioamla audio visualize`` - Generate spectrograms

**dataset** - Dataset management:

- ``bioamla dataset download`` - Download files from URL
- ``bioamla dataset unzip`` - Extract ZIP archives
- ``bioamla dataset zip`` - Create ZIP archives
- ``bioamla dataset merge`` - Merge multiple datasets
- ``bioamla dataset augment`` - Augment audio for training
- ``bioamla dataset license`` - Generate attribution files

**models** - ML model operations:

- ``bioamla models list`` - List available model types
- ``bioamla models info`` - Display model information
- ``bioamla models embed`` - Extract audio embeddings
- ``bioamla models convert`` - Convert model formats (PyTorch to ONNX)
- ``bioamla models ensemble`` - Create model ensemble

**models predict** - Run inference:

- ``bioamla models predict ast`` - AST model inference
- ``bioamla models predict generic`` - Generic model inference (multi-model)

**models train** - Train models:

- ``bioamla models train ast`` - Fine-tune AST model
- ``bioamla models train cnn`` - Train CNN model (ResNet)
- ``bioamla models train spec`` - Train spectrogram classifier

**models evaluate** - Evaluate models:

- ``bioamla models evaluate ast`` - Evaluate AST model

**detect** - Detection algorithms:

- ``bioamla detect energy`` - Band-limited energy detection
- ``bioamla detect ribbit`` - RIBBIT periodic call detection
- ``bioamla detect peaks`` - CWT peak detection
- ``bioamla detect batch`` - Batch detection

**cluster** - Clustering and discovery:

- ``bioamla cluster reduce`` - Dimensionality reduction
- ``bioamla cluster cluster`` - Cluster embeddings
- ``bioamla cluster novelty`` - Detect novel sounds

**learn** - Active learning:

- ``bioamla learn init`` - Initialize active learning
- ``bioamla learn query`` - Query samples for annotation
- ``bioamla learn annotate`` - Add annotations
- ``bioamla learn status`` - Show learning status

**indices** - Acoustic indices:

- ``bioamla indices compute`` - Compute all indices
- ``bioamla indices aci`` - Acoustic Complexity Index
- ``bioamla indices adi`` - Acoustic Diversity Index
- ``bioamla indices ndsi`` - Normalized Difference Soundscape Index

**services** - External service integrations:

- ``bioamla services clear-cache`` - Clear API response caches

**services xc** - Xeno-canto:

- ``bioamla services xc search`` - Search Xeno-canto
- ``bioamla services xc download`` - Download from Xeno-canto

**services ml** - Macaulay Library:

- ``bioamla services ml search`` - Search Macaulay Library
- ``bioamla services ml download`` - Download from Macaulay Library

**services species** - Species lookup:

- ``bioamla services species lookup`` - Look up species names
- ``bioamla services species search`` - Search for species

**services inat** - iNaturalist:

- ``bioamla services inat download`` - Download iNaturalist audio
- ``bioamla services inat search`` - Search observations
- ``bioamla services inat stats`` - Show statistics

**services hf** - HuggingFace Hub:

- ``bioamla services hf push-model`` - Push model to Hub
- ``bioamla services hf push-dataset`` - Push dataset to Hub

**services ebird** - eBird:

- ``bioamla services ebird validate`` - Validate species for location
- ``bioamla services ebird nearby`` - Get nearby observations

**services pg** - PostgreSQL:

- ``bioamla services pg export`` - Export detections to PostgreSQL
- ``bioamla services pg stats`` - Show database statistics

API Reference
-------------

.. automodule:: bioamla.cli
   :members:
   :undoc-members:
   :show-inheritance:
