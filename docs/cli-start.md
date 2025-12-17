# Getting Started with Bioamla CLI

This guide covers command-line usage of bioamla for bioacoustic analysis and machine learning workflows.

## Installation

```bash
pip install bioamla
```

For GPU support, install PyTorch with CUDA first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bioamla
```

Verify installation:

```bash
bioamla version
bioamla devices  # Check GPU availability
```

## Basic Inference

### Single File Classification

```bash
bioamla models predict ast recording.wav
```

### Directory Processing

```bash
# Process all audio files in a directory
bioamla models predict ast ./audio/

# Save results to CSV
bioamla models predict ast ./audio/ --output results.csv

# Customize prediction parameters
bioamla models predict ast ./audio/ --top-k 10 --min-confidence 0.1
```

### Batch Processing Options

```bash
# Recursive directory search
bioamla models predict ast ./data/ --recursive

# Segment long recordings
bioamla models predict ast ./audio/ --clip-seconds 10 --overlap-seconds 2

# Adjust batch size for memory management
bioamla models predict ast ./audio/ --batch-size 4
```

## Audio Processing

### Get Audio Information

```bash
# Single file info
bioamla audio info recording.wav

# List files in directory
bioamla audio list ./recordings/
```

### Generate Spectrograms

```bash
# Basic spectrogram
bioamla audio visualize recording.wav -o spectrogram.png

# Different visualization types
bioamla audio visualize recording.wav -o spec.png --type mel
bioamla audio visualize recording.wav -o spec.png --type stft
bioamla audio visualize recording.wav -o spec.png --type mfcc
bioamla audio visualize recording.wav -o spec.png --type waveform

# Batch processing
bioamla audio visualize ./audio/ -o ./spectrograms/
```

### Signal Processing

```bash
# Bandpass filter
bioamla audio filter --bandpass --low 500 --high 8000 ./audio/

# Lowpass and highpass
bioamla audio filter --lowpass --cutoff 4000 ./audio/
bioamla audio filter --highpass --cutoff 200 ./audio/

# Denoise audio
bioamla audio denoise --method spectral ./audio/

# Normalize levels
bioamla audio normalize --target-db -20 ./audio/

# Resample
bioamla audio resample --rate 16000 ./audio/

# Segment on silence
bioamla audio segment --silence-threshold -40 ./audio/
```

## Project System

Projects provide organized workflows with persistent configuration and command history.

### Creating a Project

```bash
# Initialize in current directory
bioamla project init

# With custom name
bioamla project init -n "Frog Species Study"

# Using a template
bioamla project init -n "My Study" -t research
```

### Project Templates

| Template | Description |
|----------|-------------|
| `default` | Balanced settings for general use |
| `minimal` | Sparse config, relies on defaults |
| `research` | DEBUG logging, high-resolution outputs |
| `production` | Optimized for batch processing |

### Working with Projects

```bash
# Check project status
bioamla project status

# View project configuration
bioamla project config show

# Edit configuration interactively
bioamla project config edit
```

### Project Structure

After initialization, your project directory contains:

```text
my-project/
├── .bioamla/
│   ├── config.toml      # Project configuration
│   └── history.db       # Command history
├── data/                # Input data
├── output/              # Results
└── models/              # Saved models
```

### Command History

Projects track all commands for reproducibility:

```bash
# View recent commands
bioamla log show

# Show more history
bioamla log show --limit 50

# Search command history
bioamla log search "predict"

# View usage statistics
bioamla log stats
```

## Configuration

### Configuration Hierarchy

Settings are loaded in priority order (highest to lowest):

1. CLI flags (`--sample-rate 22050`)
2. Explicit `--config` option
3. Project config (`.bioamla/config.toml`)
4. Current directory (`./bioamla.toml`)
5. User config (`~/.config/bioamla/config.toml`)
6. System config (`/etc/bioamla/config.toml`)
7. Built-in defaults

### Managing Configuration

```bash
# Show current configuration
bioamla config show

# Create default config file
bioamla config init

# Show config search paths
bioamla config path

# Clear model cache
bioamla config purge --models

# Clear all caches
bioamla config purge --all
```

### Example Configuration File

Create `bioamla.toml` in your project directory:

```toml
[project]
name = "my-study"
description = "Species identification project"

[audio]
sample_rate = 16000
mono = true
normalize = false

[visualize]
type = "mel"
n_fft = 2048
hop_length = 512
n_mels = 128
cmap = "magma"

[models]
default_ast_model = "MIT/ast-finetuned-audioset-10-10-0.4593"

[inference]
batch_size = 8
top_k = 5
min_confidence = 0.01
clip_seconds = 10
overlap_seconds = 0

[training]
learning_rate = 5.0e-5
epochs = 10
batch_size = 16

[batch]
recursive = true
workers = 1

[output]
format = "csv"
overwrite = false

[progress]
enabled = true
style = "rich"

[logging]
level = "WARNING"
```

## Model Training

### Fine-tune on Custom Data

Prepare your data in this structure:

```text
training_data/
├── train/
│   ├── species_a/
│   │   ├── recording1.wav
│   │   └── recording2.wav
│   └── species_b/
│       └── recording3.wav
└── val/
    ├── species_a/
    │   └── recording4.wav
    └── species_b/
        └── recording5.wav
```

Train the model:

```bash
bioamla models train ast \
  --train-dir ./data/train \
  --val-dir ./data/val \
  --output-dir ./models/finetuned \
  --epochs 10 \
  --learning-rate 5e-5
```

### Model Management

```bash
# List available models
bioamla models list

# Get model info
bioamla models info --model-path ./models/finetuned

# Extract embeddings
bioamla models embed --input audio.wav --output embeddings.npy

# Create ensemble predictions
bioamla models ensemble --model-dirs ./m1 ./m2 --strategy voting
```

## Detection Commands

### Energy-Based Detection

```bash
# Detect sounds in frequency band
bioamla detect energy --low-freq 500 --high-freq 3000 ./audio/
```

### Specialized Detectors

```bash
# Periodic call detection (e.g., frog calls)
bioamla detect ribbit --pulse-rate 10 ./audio/

# Peak detection
bioamla detect peaks --snr 10 ./audio/

# Accelerating pattern detection
bioamla detect accelerating --min-pulses 3 ./audio/
```

## Acoustic Indices

Calculate soundscape metrics:

```bash
# Compute all indices
bioamla indices compute ./audio/

# Individual indices
bioamla indices aci ./audio/    # Acoustic Complexity Index
bioamla indices adi ./audio/    # Acoustic Diversity Index
bioamla indices aei ./audio/    # Acoustic Evenness Index
bioamla indices bio ./audio/    # Bioacoustic Index

# NDSI with custom frequency bands
bioamla indices ndsi --anthro-min 1000 --anthro-max 2000 ./audio/
```

## Clustering & Analysis

```bash
# Dimensionality reduction
bioamla cluster reduce --embeddings emb.npy --method umap --output reduced.npy

# Clustering
bioamla cluster cluster --embeddings emb.npy --method hdbscan --output labels.npy

# Novelty detection
bioamla cluster novelty --embeddings emb.npy --output novel.npy
```

## Active Learning

Interactive annotation workflow:

```bash
# Initialize active learning session
bioamla learn init --predictions predictions.csv

# Query samples needing annotation
bioamla learn query --n-samples 20

# Add annotations
bioamla learn annotate --annotations new_labels.csv

# Check status
bioamla learn status
```

## External Data Services

### iNaturalist

```bash
# Search for recordings
bioamla services inat search --species "Lithobates catesbeianus"

# Download recordings
bioamla services inat download --species "Lithobates catesbeianus" --output-dir ./data
```

### Xeno-canto

```bash
# Search bird recordings
bioamla services xc search --species "Turdus migratorius"

# Download recordings
bioamla services xc download --species "Turdus migratorius"
```

### eBird

```bash
# Get nearby observations
bioamla services ebird nearby --lat 40.7128 --lng -74.0060

# Validate species code
bioamla services ebird validate --species-code "amerob"
```

## Real-time Audio

```bash
# List audio input devices
bioamla realtime devices

# Test recording
bioamla realtime test --duration 5
```

## Utility Commands

```bash
# Show CUDA/GPU information
bioamla devices

# Show version
bioamla version

# Explore dataset structure
bioamla explore ./dataset/

# Get help
bioamla --help
bioamla models --help
bioamla models predict --help
```

## Example Workflows

### Quick Species Survey

```bash
# Create project
bioamla project init -n "Field Survey 2025"

# Process recordings
bioamla models predict ast ./recordings/ -o results.csv

# Generate spectrograms for review
bioamla audio visualize ./recordings/ -o ./spectrograms/
```

### Research Pipeline

```bash
# Initialize research project
bioamla project init -n "Frog Study" -t research

# Compute acoustic indices
bioamla indices compute ./audio/ -o indices.csv

# Extract embeddings for analysis
bioamla models embed --input ./audio/ --output embeddings.npy

# Cluster similar sounds
bioamla cluster cluster --embeddings embeddings.npy --method hdbscan -o clusters.npy
```

### Model Development

```bash
# Download training data
bioamla services xc download --species "Lithobates catesbeianus" -o ./data/train/bullfrog
bioamla services xc download --species "Lithobates clamitans" -o ./data/train/greenfrog

# Train model
bioamla models train ast --train-dir ./data/train --epochs 20 -o ./models/frog_classifier

# Evaluate on new recordings
bioamla models predict ast --model-path ./models/frog_classifier ./test_recordings/
```

## Getting Help

```bash
# General help
bioamla --help

# Command group help
bioamla models --help
bioamla audio --help
bioamla project --help

# Specific command help
bioamla models predict --help
bioamla audio visualize --help
```
