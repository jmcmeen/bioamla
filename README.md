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

### Installation w/ pip

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
bioamla ast predict path/to/audio.wav --model-path bioamla/scp-frogs
```

This will output the predicted class for the audio file.

### 2. Batch Inference on Directory

Process all audio files in a directory and export results to CSV using the `--batch` flag:

```bash
bioamla ast predict /path/to/audio/directory --batch \
  --output-csv results.csv \
  --model-path bioamla/scp-frogs \
  --resample-freq 16000 \
  --clip-seconds 1 \
  --overlap-seconds 0.5
```

This creates a CSV file with columns: `filepath`, `start`, `stop`, `prediction`

**Optimized inference with GPU acceleration:**

```bash
bioamla ast predict /path/to/audio/directory --batch \
  --model-path bioamla/scp-frogs \
  --batch-size 16 \
  --fp16 \
  --compile \
  --workers 4
```

Performance options (batch mode only):

- `--batch-size`: Process multiple segments in one forward pass (2-4x faster)
- `--fp16`: Use half-precision inference (~2x faster on modern GPUs)
- `--compile`: Use torch.compile() for optimized execution (1.5-2x faster, PyTorch 2.0+)
- `--workers`: Parallel file loading for I/O-bound workloads

**Resume interrupted processing:**

```bash
bioamla ast predict /path/to/audio/directory --batch \
  --output-csv results.csv \
  --no-restart
```

### 3. Fine-tune a Model

Train a custom model on your own dataset from Hugging Face Hub:

```bash
bioamla ast train \
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
bioamla ast train \
  --training-dir ./my-training \
  --train-dataset your-username/your-dataset \
  --num-train-epochs 10 \
  --push-to-hub
```

### 4. Model Evaluation

Evaluate model performance on a test dataset with ground truth labels:

```bash
bioamla ast evaluate ./test_audio --model-path bioamla/scp-frogs \
  --ground-truth labels.csv
```

This outputs:

- Accuracy, Precision, Recall, F1 Score
- Per-class metrics
- Confusion matrix

**Save results to different formats:**

```bash
# JSON output
bioamla ast evaluate ./test_audio --ground-truth labels.csv \
  --output results.json --format json

# CSV output (per-class metrics)
bioamla ast evaluate ./test_audio --ground-truth labels.csv \
  --output results.csv --format csv
```

### 5. Spectrogram Visualization

Generate spectrograms and other audio visualizations:

**Single file:**

```bash
bioamla visualize audio.wav --output spectrogram.png
```

**Batch processing:**

```bash
bioamla visualize ./audio_dir --batch --output ./spectrograms
```

**Visualization types:**

```bash
# Mel spectrogram (default)
bioamla visualize audio.wav --type mel --output mel_spec.png

# MFCC visualization
bioamla visualize audio.wav --type mfcc --output mfcc.png

# Waveform plot
bioamla visualize audio.wav --type waveform --output waveform.png
```

### 6. Audio Augmentation

Expand training datasets with augmented audio:

```bash
bioamla augment ./audio --output ./augmented \
  --add-noise 3-30 \
  --time-stretch 0.8-1.2 \
  --pitch-shift -2,2 \
  --multiply 5
```

This creates 5 augmented copies of each audio file with random combinations of:

- Gaussian noise (SNR 3-30 dB)
- Time stretching (80%-120% speed)
- Pitch shifting (-2 to +2 semitones)

### 7. Signal Processing

Process audio files with filtering, denoising, and other operations:

**Apply frequency filters:**

```bash
# Bandpass filter (keep 1000-8000 Hz)
bioamla audio filter recording.wav --bandpass 1000-8000 --output filtered.wav

# Lowpass filter (remove high frequencies)
bioamla audio filter recording.wav --lowpass 4000 --output lowpassed.wav

# Highpass filter (remove low frequencies)
bioamla audio filter recording.wav --highpass 500 --output highpassed.wav
```

**Remove noise:**

```bash
bioamla audio denoise noisy.wav --output clean.wav --strength 1.5
```

**Split audio on silence:**

```bash
bioamla audio segment long_recording.wav --output ./segments \
  --silence-threshold -40 \
  --min-silence 0.3 \
  --min-segment 0.5
```

**Detect onset events:**

```bash
bioamla audio detect-events recording.wav --output events.csv
```

**Normalize loudness:**

```bash
# RMS normalization (default)
bioamla audio normalize recording.wav --target-db -20 --output normalized.wav

# Peak normalization
bioamla audio normalize recording.wav --peak --target-db -3 --output normalized.wav
```

**Resample audio:**

```bash
bioamla audio resample recording.wav --rate 16000 --output resampled.wav
```

**Trim audio:**

```bash
# Trim by time
bioamla audio trim recording.wav --start 1.5 --end 5.0 --output trimmed.wav

# Trim silence from start and end
bioamla audio trim recording.wav --silence --threshold -40 --output trimmed.wav
```

**Batch processing:**

All signal processing commands support `--batch` for directory processing:

```bash
bioamla audio normalize ./recordings --batch --output ./normalized --target-db -20
bioamla audio filter ./recordings --batch --output ./filtered --lowpass 8000
```

### 8. Audio File Utilities

**List all audio files in a directory:**

```bash
bioamla audio list /path/to/audio/directory
```

**Extract WAV file metadata:**

```bash
bioamla audio info /path/to/file.wav
```

**Download audio files:**

```bash
bioamla download https://example.com/audio.zip ./downloads
```

**Extract archives:**

```bash
bioamla unzip ./downloads/audio.zip ./extracted
```

### 9. License File Generation

Generate license/attribution files from dataset metadata:

**Single dataset:**

```bash
bioamla dataset license ./my_dataset
```

**With a template file:**

```bash
bioamla dataset license ./my_dataset --template ./license_template.txt
```

**Process all datasets in a directory:**

```bash
bioamla dataset license ./audio_datasets --batch
```

**Custom output filename:**

```bash
bioamla dataset license ./my_dataset --output ATTRIBUTION.txt
```

The metadata CSV must contain these columns: `file_name`, `attr_id`, `attr_lic`, `attr_url`, `attr_note`

### 10. Python API Usage

Use bioamla programmatically in your Python scripts:

```python
from bioamla.ast import load_pretrained_ast_model, wav_ast_inference
from bioamla.torchaudio import load_waveform_tensor

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
from bioamla.ast import wave_file_batch_inference

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
from bioamla.torchaudio import (
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

### 11. System Diagnostics

**Check GPU availability:**

```python
from bioamla.diagnostics import get_device_info

device_info = get_device_info()
print(f"CUDA available: {device_info['cuda_available']}")
print(f"Device count: {device_info['device_count']}")
print(f"Device name: {device_info['device_name']}")
```

**Get package versions:**

```python
from bioamla.diagnostics import get_package_versions

versions = get_package_versions()
for package, version in versions.items():
    print(f"{package}: {version}")
```

### 12. Dataset Explorer (Experimental)

Launch an interactive terminal dashboard to explore audio datasets:

```bash
bioamla explore ./my_dataset
```

The explorer provides:

- File browser with sorting and filtering
- Dataset statistics (total files, size, formats)
- Category and split summaries (if metadata.csv present)
- Audio playback (requires system audio player)
- Spectrogram generation and viewing
- Search functionality

### 13. Experiment Tracking with MLflow

bioamla integrates with MLflow for experiment tracking during model training:

**Start MLflow server:**

```bash
mlflow server --host 0.0.0.0 --port 5000
```

**Train with MLflow tracking:**

```bash
bioamla ast train \
  --training-dir "my-model" \
  --train-dataset "bioamla/scp-frogs" \
  --num-train-epochs 10 \
  --mlflow-tracking-uri "http://localhost:5000" \
  --mlflow-experiment-name "frog-classifier" \
  --mlflow-run-name "baseline-run"
```

**View experiments in MLflow UI:**

Open `http://localhost:5000` in your browser to view training metrics, compare runs, and analyze model performance.

MLflow tracks:

- Training and evaluation metrics (loss, accuracy)
- Model hyperparameters
- Training artifacts

## CLI Reference

| Command | Description |
|---------|-------------|
| `bioamla version` | Display bioamla version |
| `bioamla devices` | Show CUDA/GPU information |
| `bioamla explore <DIR>` | Launch interactive TUI dashboard for exploring datasets |
| `bioamla purge` | Purge cached HuggingFace Hub data (models/datasets) |
| `bioamla visualize <PATH>` | Generate spectrogram visualizations |
| `bioamla augment <INPUT_DIR>` | Augment audio files to expand training datasets |
| `bioamla download <URL> [DIR]` | Download files from URL |
| `bioamla unzip <FILE> [DIR]` | Extract ZIP archives |
| `bioamla zip <SOURCE> <OUTPUT>` | Create ZIP archive from file or directory |

### Audio Commands (`bioamla audio`)

| Command | Description |
|---------|-------------|
| `bioamla audio list [DIR]` | List audio files in directory |
| `bioamla audio info <FILE>` | Display WAV file metadata |
| `bioamla audio convert <PATH> <FORMAT>` | Convert audio files between formats |
| `bioamla audio filter <PATH>` | Apply frequency filters (bandpass, lowpass, highpass) |
| `bioamla audio denoise <PATH>` | Apply spectral noise reduction |
| `bioamla audio segment <PATH>` | Split audio on silence into separate files |
| `bioamla audio detect-events <PATH>` | Detect onset events and export to CSV |
| `bioamla audio normalize <PATH>` | Normalize audio loudness (RMS or peak) |
| `bioamla audio resample <PATH>` | Resample audio to a different sample rate |
| `bioamla audio trim <PATH>` | Trim audio by time or remove silence |

### AST Model Commands (`bioamla ast`)

| Command | Description |
|---------|-------------|
| `bioamla ast predict <PATH>` | Single file or batch inference |
| `bioamla ast train` | Fine-tune AST model on custom datasets |
| `bioamla ast evaluate <PATH>` | Evaluate model on test data with ground truth labels |

### iNaturalist Commands (`bioamla inat`)

| Command | Description |
|---------|-------------|
| `bioamla inat download <OUTPUT_DIR>` | Download audio observations from iNaturalist |
| `bioamla inat search` | Search for taxa with observations in a place or project |
| `bioamla inat stats <PROJECT_ID>` | Get statistics for an iNaturalist project |

### Dataset Commands (`bioamla dataset`)

| Command | Description |
|---------|-------------|
| `bioamla dataset merge <OUTPUT_DIR> <PATHS...>` | Merge multiple audio datasets into one |
| `bioamla dataset license <PATH>` | Generate license/attribution file from metadata |

### HuggingFace Hub Commands (`bioamla hf`)

| Command | Description |
|---------|-------------|
| `bioamla hf push-model <PATH> <REPO_ID>` | Push model folder to HuggingFace Hub |
| `bioamla hf push-dataset <PATH> <REPO_ID>` | Push dataset folder to HuggingFace Hub |

Both push commands automatically detect large folders (>5GB or >1000 files) and use the optimized `upload_large_folder` method for better reliability.

Use `bioamla <command> --help` for detailed options on any command.

## Technologies

- **PyTorch + HuggingFace Transformers**: Audio Spectrogram Transformer models
- **TorchAudio**: Audio file I/O and preprocessing
- **Librosa**: Audio analysis and feature extraction
- **SciPy**: Signal processing and filtering
- **Click**: Command-line interface framework
- **Textual**: Terminal user interface for dataset exploration
- **FastAPI**: Web service capability (optional)
- **Pydantic**: Data validation and API schemas
- **Audiomentations**: Audio data augmentation
- **Matplotlib**: Spectrogram visualization
- **TensorBoard**: Training visualization
- **MLflow**: Experiment tracking and model management (optional)

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
