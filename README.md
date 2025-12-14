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

### 4. Spectrogram Visualization

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

### 5. Audio Augmentation

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

### 6. Audio File Utilities

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

### 7. Python API Usage

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

### 8. System Diagnostics

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

### 9. Experiment Tracking with MLflow

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

## CLI Commands Reference

### System Commands

| Command | Description |
|---------|-------------|
| `bioamla version` | Display bioamla version |
| `bioamla devices` | Show CUDA/GPU information |
| `bioamla purge` | Purge cached HuggingFace Hub data (models/datasets) |

**Purge cached data:**

```bash
bioamla purge --models          # Purge only cached models
bioamla purge --datasets        # Purge only cached datasets
bioamla purge --all             # Purge everything
bioamla purge --all -y          # Purge everything without confirmation
```

### Audio Commands (`bioamla audio`)

| Command | Description |
|---------|-------------|
| `bioamla audio list [DIR]` | List audio files in directory |
| `bioamla audio info <FILE>` | Display WAV file metadata |
| `bioamla audio convert <DATASET_PATH> <FORMAT>` | Convert all audio files in a dataset |

**audio convert options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--metadata-filename` | `metadata.csv` | Name of metadata CSV file |
| `--keep-original` | | Keep original files after conversion |
| `--quiet` | | Suppress progress output |

Supported formats: wav, mp3, m4a, aac, flac, ogg, wma

### Visualization Commands

| Command | Description |
|---------|-------------|
| `bioamla visualize <PATH>` | Generate spectrogram visualizations (use `--batch` for directories) |

**visualize options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | | Output file path (single file) or directory (batch mode) |
| `--batch` | | Process all audio files in a directory |
| `--type` | `mel` | Visualization type: `mel`, `mfcc`, or `waveform` |
| `--sample-rate` | `16000` | Target sample rate for processing |
| `--n-mels` | `128` | Number of mel bands (mel spectrogram only) |
| `--n-mfcc` | `40` | Number of MFCCs (mfcc only) |
| `--cmap` | `magma` | Colormap for spectrogram visualizations |
| `--recursive/--no-recursive` | `--recursive` | Search subdirectories (batch mode only) |
| `--quiet` | | Suppress progress output |

### File Utilities

| Command | Description |
|---------|-------------|
| `bioamla download <URL> [DIR]` | Download files from URL |
| `bioamla unzip <FILE> [DIR]` | Extract ZIP archives |
| `bioamla zip <SOURCE> <OUTPUT>` | Create ZIP archive from file or directory |

### AST Model Commands (`bioamla ast`)

| Command | Description |
|---------|-------------|
| `bioamla ast predict <PATH>` | Single file or batch inference (use `--batch` for directories) |
| `bioamla ast train` | Fine-tune AST model on custom datasets |
| `bioamla ast push <MODEL_PATH> <REPO_ID>` | Push fine-tuned model to HuggingFace Hub |

**ast predict options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | `bioamla/scp-frogs` | AST model to use for inference |
| `--resample-freq` | `16000` | Resampling frequency |
| `--batch` | | Run batch inference on a directory of audio files |
| `--output-csv` | `output.csv` | Output CSV file name (batch mode only) |
| `--clip-seconds` | `1` | Duration of audio clips in seconds (batch mode only) |
| `--overlap-seconds` | `0` | Overlap between clips in seconds (batch mode only) |
| `--restart/--no-restart` | `--no-restart` | Resume from existing results (batch mode only) |
| `--batch-size` | `8` | Number of segments to process in parallel (batch mode only) |
| `--fp16/--no-fp16` | `--no-fp16` | Use half-precision inference (batch mode only) |
| `--compile/--no-compile` | `--no-compile` | Use torch.compile() for optimized inference (batch mode only) |
| `--workers` | `1` | Number of parallel workers for file loading (batch mode only) |

**ast train options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--training-dir` | `.` | Directory to save training outputs |
| `--base-model` | `MIT/ast-finetuned-audioset-10-10-0.4593` | Base model to fine-tune |
| `--train-dataset` | `bioamla/scp-frogs` | Training dataset from HuggingFace Hub |
| `--split` | `train` | Dataset split to use |
| `--category-id-column` | `target` | Column name for category IDs |
| `--category-label-column` | `category` | Column name for category labels |
| `--report-to` | `tensorboard` | Where to report metrics |
| `--learning-rate` | `5e-5` | Learning rate for training |
| `--push-to-hub/--no-push-to-hub` | `--no-push-to-hub` | Push model to HuggingFace Hub |
| `--num-train-epochs` | `1` | Number of training epochs |
| `--per-device-train-batch-size` | `8` | Training batch size per device |
| `--eval-strategy` | `epoch` | Evaluation strategy |
| `--save-strategy` | `epoch` | Model save strategy |
| `--eval-steps` | `1` | Steps between evaluations |
| `--save-steps` | `1` | Steps between saves |
| `--load-best-model-at-end/--no-load-best-model-at-end` | `--load-best-model-at-end` | Load best model at end |
| `--metric-for-best-model` | `accuracy` | Metric for best model selection |
| `--logging-strategy` | `steps` | Logging strategy |
| `--logging-steps` | `100` | Steps between logging |
| `--fp16/--no-fp16` | `--no-fp16` | Use FP16 mixed precision training (NVIDIA GPUs) |
| `--bf16/--no-bf16` | `--no-bf16` | Use BF16 mixed precision training (Ampere+ GPUs) |
| `--gradient-accumulation-steps` | `1` | Number of gradient accumulation steps |
| `--dataloader-num-workers` | `4` | Number of dataloader workers |
| `--torch-compile/--no-torch-compile` | `--no-torch-compile` | Use torch.compile for faster training (PyTorch 2.0+) |
| `--finetune-mode` | `full` | Training mode: `full` (all layers) or `feature-extraction` (freeze base, train classifier only) |
| `--mlflow-tracking-uri` | | MLflow tracking server URI (e.g., `http://localhost:5000`) |
| `--mlflow-experiment-name` | | MLflow experiment name |
| `--mlflow-run-name` | | MLflow run name |

**MLflow integration example:**

```bash
# Start MLflow server (in separate terminal)
mlflow server --host 0.0.0.0 --port 5000

# Train with MLflow tracking
bioamla ast train \
  --training-dir "my-model" \
  --train-dataset "bioamla/scp-frogs" \
  --num-train-epochs 10 \
  --mlflow-tracking-uri "http://localhost:5000" \
  --mlflow-experiment-name "frog-classifier" \
  --mlflow-run-name "baseline-run"
```

**ast push options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--private/--public` | `--public` | Make the repository private or public |
| `--commit-message` | | Custom commit message for the push |

**Push model to HuggingFace Hub:**

```bash
# Push a fine-tuned model to HuggingFace Hub
bioamla ast push ./my-training/best_model myusername/my-frog-classifier

# Push as a private repository
bioamla ast push ./my-model myusername/private-model --private

# Push with a custom commit message
bioamla ast push ./my-model myusername/my-model --commit-message "v1.0 release"
```

Note: You must be logged in to HuggingFace Hub first using `huggingface-cli login`.

### iNaturalist Commands (`bioamla inat`)

| Command | Description |
|---------|-------------|
| `bioamla inat download <OUTPUT_DIR>` | Download audio observations from iNaturalist |
| `bioamla inat search` | Search for taxa with observations in a place or project |
| `bioamla inat stats <PROJECT_ID>` | Get statistics for an iNaturalist project |

**inat download options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--taxon-ids` | | Comma-separated list of taxon IDs |
| `--taxon-csv` | | Path to CSV file with taxon_id column |
| `--taxon-name` | | Filter by taxon name (e.g., "Aves") |
| `--place-id` | | Filter by place ID (e.g., 1 for US) |
| `--user-id` | | Filter by observer username |
| `--project-id` | | Filter by iNaturalist project ID or slug |
| `--quality-grade` | `research` | Quality grade: research, needs_id, casual |
| `--sound-license` | | Comma-separated list of licenses |
| `--start-date` | | Start date (YYYY-MM-DD) |
| `--end-date` | | End date (YYYY-MM-DD) |
| `--obs-per-taxon` | `100` | Observations to download per taxon ID |
| `--organize-by-taxon/--no-organize-by-taxon` | `--organize-by-taxon` | Organize into subdirectories by species |
| `--include-inat-metadata` | | Include additional iNaturalist metadata |
| `--file-extensions` | | Comma-separated list of extensions to filter |
| `--delay` | `1.0` | Delay between downloads (rate limiting) |
| `--quiet` | | Suppress progress output |

**inat search options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--place-id` | | Filter by place ID |
| `--project-id` | | Filter by project ID or slug |
| `--taxon-id` | | Filter by parent taxon ID |
| `--quality-grade` | `research` | Quality grade filter |
| `--output, -o` | | Output file path for CSV |
| `--quiet` | | Suppress progress output |

**inat stats options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | | Output file path for JSON |
| `--quiet` | | Suppress output, print only JSON |

**Download audio from a CSV of taxon IDs:**

```bash
# First, search for taxa and export to CSV
bioamla inat search --project-id appalachia-bioacoustics --taxon-id 20979 -o taxa.csv

# Then download audio for all taxa in the CSV
bioamla inat download ./sounds --taxon-csv taxa.csv --obs-per-taxon 10
```

The CSV file should have a `taxon_id` column with integer taxon IDs:

```csv
taxon_id,name,common_name,observation_count
65489,Lithobates catesbeianus,American Bullfrog,150
23456,Anaxyrus americanus,American Toad,200
```

### Dataset Commands (`bioamla dataset`)

| Command | Description |
|---------|-------------|
| `bioamla dataset merge <OUTPUT_DIR> <PATHS...>` | Merge multiple audio datasets into one |

**dataset merge options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--metadata-filename` | `metadata.csv` | Name of metadata CSV file in each dataset |
| `--overwrite` | | Overwrite existing files instead of skipping |
| `--no-organize` | | Preserve original directory structure |
| `--target-format` | | Convert all audio files to this format |
| `--quiet` | | Suppress progress output |

## Technologies

- **PyTorch + HuggingFace Transformers**: Audio Spectrogram Transformer models
- **TorchAudio**: Audio file I/O and preprocessing
- **Click**: Command-line interface framework
- **FastAPI**: Web service capability (optional)
- **Pydantic**: Data validation and API schemas
- **Audiomentations**: Audio data augmentation
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
