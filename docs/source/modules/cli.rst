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
- ``bioamla models predict`` - Run inference with ML model
- ``bioamla models embed`` - Extract audio embeddings
- ``bioamla models train`` - Train custom CNN model
- ``bioamla models ast-predict`` - AST model inference
- ``bioamla models ast-train`` - Fine-tune AST model
- ``bioamla models ast-evaluate`` - Evaluate AST model

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

**api** - External API integrations:

- ``bioamla api xc-search`` - Search Xeno-canto
- ``bioamla api xc-download`` - Download from Xeno-canto
- ``bioamla api ml-search`` - Search Macaulay Library
- ``bioamla api species`` - Species name lookup

**inat** - iNaturalist integration:

- ``bioamla inat download`` - Download iNaturalist audio
- ``bioamla inat search`` - Search observations
- ``bioamla inat stats`` - Show statistics

**hf** - HuggingFace Hub:

- ``bioamla hf push-model`` - Push model to Hub
- ``bioamla hf push-dataset`` - Push dataset to Hub

**integrations** - External integrations:

- ``bioamla integrations ebird-validate`` - Validate against eBird
- ``bioamla integrations pg-export`` - Export to PostgreSQL

API Reference
-------------

.. automodule:: bioamla.cli
   :members:
   :undoc-members:
   :show-inheritance:
