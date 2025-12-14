# Bioacoustics and Machine Learning Applications (bioamla)

A Python package for audio analysis and machine learning-based audio classification, focusing on bioacoustic data. Bioamla specializes in wildlife sound analysis using Audio Spectrogram Transformer (AST) models.

> **Prerelease Notice:** This is a prerelease version of bioamla. The package is functional and ready for use, but additional features, improvements, and documentation updates are planned for 2026.

## Description

Bioamla provides a toolkit for researchers, biologists, and machine learning engineers working with environmental sound data and species identification. The package combines robust audio processing capabilities with deep learning models to enable:

- **Audio Classification**: Classify wildlife sounds, species calls, and environmental audio using pre-trained or fine-tuned AST models
- **Model Training**: Fine-tune Audio Spectrogram Transformer models on custom datasets from Hugging Face Hub
- **Batch Processing**: Efficiently process directories of audio files with temporal segmentation
- **Audio Processing**: Load, resample, split, and extract metadata from various audio formats
- **System Diagnostics**: Monitor GPU/CUDA availability and package versions

## Setup

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training and large-scale inference)

### Installation

Install bioamla using pip:

```bash
pip install bioamla
```

### Installation from Source

For development or the latest version:

```bash
git clone https://github.com/jmcmeen/bioamla.git
cd bioamla
pip install -e .
```

### Verify Installation

Check that bioamla is installed correctly:

```bash
bioamla version
bioamla devices
```

The `devices` command will show your CUDA/GPU availability and configuration.

## Examples

### 1. Basic Audio Classification

Classify a single audio file using a pre-trained model:

```bash
bioamla ast-predict path/to/audio.wav bioamla/scp-frogs 16000
```

This will output the top predictions with confidence scores for the audio file.

### 2. Batch Inference on Directory

Process all audio files in a directory and export results to CSV:

```bash
bioamla ast-batch-inference /path/to/audio/directory \
  --output-csv results.csv \
  --model-path bioamla/scp-frogs \
  --resample-freq 16000 \
  --clip-seconds 1 \
  --overlap-seconds 0.5
```

This creates a CSV file with columns: `filepath`, `start`, `stop`, `prediction`

**Resume interrupted processing:**

```bash
bioamla ast-batch-inference /path/to/audio/directory \
  --output-csv results.csv \
  --no-restart
```

### 3. Fine-tune a Model

Train a custom model on your own dataset from Hugging Face Hub:

```bash
bioamla ast-finetune \
  --training-dir ./my-training \
  --base-model MIT/ast-finetuned-audioset-10-10-0.4593 \
  --train-dataset your-username/your-dataset \
  --num-train-epochs 10 \
  --learning-rate 5e-5 \
  --per-device-train-batch-size 8 \
  --eval-strategy epoch \
  --report-to tensorboard
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./my-training
```

**Push trained model to Hugging Face Hub:**

```bash
bioamla ast-finetune \
  --training-dir ./my-training \
  --train-dataset your-username/your-dataset \
  --num-train-epochs 10 \
  --push-to-hub
```

### 4. Audio File Utilities

**List all audio files in a directory:**

```bash
bioamla audio /path/to/audio/directory
```

**Extract WAV file metadata:**

```bash
bioamla wave /path/to/file.wav
```

**Download audio files:**

```bash
bioamla download https://example.com/audio.zip ./downloads
```

**Extract archives:**

```bash
bioamla unzip ./downloads/audio.zip ./extracted
```

### 5. Python API Usage

Use bioamla programmatically in your Python scripts:

```python
from bioamla.core.ast import load_pretrained_ast_model, wav_ast_inference
from bioamla.core.torchaudio import load_waveform_tensor

# Load a pre-trained model
model, processor = load_pretrained_ast_model("bioamla/scp-frogs")

# Run inference on a single file
predictions = wav_ast_inference(
    wav_filepath="path/to/audio.wav",
    model_path="bioamla/scp-frogs",
    resample_freq=16000
)

# Print top predictions
for pred in predictions[:5]:
    print(f"{pred['label']}: {pred['score']:.4f}")
```

**Batch processing with segmentation:**

```python
from bioamla.core.ast import wave_file_batch_inference

# Process directory with 1-second segments and 0.5s overlap
wave_file_batch_inference(
    directory="./audio_files",
    model_path="bioamla/scp-frogs",
    output_csv="results.csv",
    resample_freq=16000,
    clip_seconds=1,
    overlap_seconds=0.5,
    restart=True
)
```

**Load and process audio:**

```python
from bioamla.core.torchaudio import (
    load_waveform_tensor,
    resample_waveform_tensor,
    split_waveform_tensor
)

# Load audio file
waveform, sample_rate = load_waveform_tensor("audio.wav")

# Resample to 16kHz
waveform_resampled = resample_waveform_tensor(
    waveform, sample_rate, 16000
)

# Split into 1-second segments with 0.5s overlap
segments = split_waveform_tensor(
    waveform_resampled,
    sample_rate=16000,
    clip_seconds=1,
    overlap_seconds=0.5
)
```

### 6. System Diagnostics

**Check GPU availability:**

```python
from bioamla.core.diagnostics import get_device_info

device_info = get_device_info()
print(f"CUDA available: {device_info['cuda_available']}")
print(f"Device count: {device_info['device_count']}")
print(f"Device name: {device_info['device_name']}")
```

**Get package versions:**

```python
from bioamla.core.diagnostics import get_package_versions

versions = get_package_versions()
for package, version in versions.items():
    print(f"{package}: {version}")
```

## CLI Commands Reference

### System Commands

| Command | Description |
|---------|-------------|
| `bioamla version` | Display bioamla version |
| `bioamla devices` | Show CUDA/GPU information |

### Audio Utilities

| Command | Description |
|---------|-------------|
| `bioamla audio [DIR]` | List audio files in directory |
| `bioamla wave <FILE>` | Display WAV file metadata |
| `bioamla download <URL> [DIR]` | Download files from URL |
| `bioamla unzip <FILE> [DIR]` | Extract ZIP archives |
| `bioamla zip <SOURCE> <OUTPUT>` | Create ZIP archive from file or directory |

### AST Model Commands

| Command | Description |
|---------|-------------|
| `bioamla ast-predict <FILE> <MODEL> <SR>` | Single file inference |
| `bioamla ast-batch-inference <DIR>` | Batch directory inference with segmentation |
| `bioamla ast-finetune` | Fine-tune AST model on custom datasets |

### iNaturalist Integration

| Command | Description |
|---------|-------------|
| `bioamla inat-audio <OUTPUT_DIR>` | Download audio observations from iNaturalist |
| `bioamla inat-taxa-search` | Search for taxa with observations in a place or project |
| `bioamla inat-project-stats <PROJECT_ID>` | Get statistics for an iNaturalist project |

**Download audio from a CSV of taxon IDs:**

```bash
# First, search for taxa and export to CSV
bioamla inat-taxa-search --project-id appalachia-bioacoustics --taxon-id 20979 -o taxa.csv

# Then download audio for all taxa in the CSV
bioamla inat-audio ./sounds --taxon-csv taxa.csv --obs-per-taxon 10
```

The CSV file should have a `taxon_id` column with integer taxon IDs:

```csv
taxon_id,name,common_name,observation_count
65489,Lithobates catesbeianus,American Bullfrog,150
23456,Anaxyrus americanus,American Toad,200
```

### Dataset Management

| Command | Description |
|---------|-------------|
| `bioamla merge-datasets <OUTPUT_DIR> <PATHS...>` | Merge multiple audio datasets into one |
| `bioamla convert-audio <DATASET_PATH> <FORMAT>` | Convert all audio files in a dataset to a specified format |

## Technologies

- **PyTorch + HuggingFace Transformers**: Audio Spectrogram Transformer models
- **TorchAudio**: Audio file I/O and preprocessing
- **Click**: Command-line interface framework
- **FastAPI**: Web service capability (optional)
- **Pydantic**: Data validation and API schemas
- **Audiomentations**: Audio data augmentation
- **TensorBoard**: Training visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use bioamla in your research, please cite:

```bibtex
@software{bioamla,
  author = {McMeen, John},
  title = {Bioamla: Bioacoustics and Machine Learning Applications},
  year = {2025},
  url = {https://github.com/jmcmeen/bioamla}
}
```
