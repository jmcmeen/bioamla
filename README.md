# bioamla

A Python CLI and library for Bioacoustics and Machine Learning Applications.

> **Prerelease Notice:** This is a prerelease version. Do not use on production data.

## Installation

### System Dependencies

Bioamla requires some system-level libraries for full functionality. Install them using the provided script:

```bash
# Download and run the installer (or clone the repo first)
curl -fsSL https://raw.githubusercontent.com/jmcmeen/bioamla/main/scripts/install-deps.sh | bash

# Or if you have the repo cloned:
./scripts/install-deps.sh

# Check what's installed:
./scripts/install-deps.sh --check
```

**Manual installation by platform:**

| Platform | Command |
|----------|---------|
| Ubuntu/Debian | `sudo apt install ffmpeg libsndfile1 portaudio19-dev` |
| Fedora | `sudo dnf install ffmpeg libsndfile portaudio` |
| Arch Linux | `sudo pacman -S ffmpeg libsndfile portaudio` |
| macOS | `brew install ffmpeg libsndfile portaudio` |

| Dependency | Purpose | Required For |
|------------|---------|--------------|
| FFmpeg | Audio format conversion | MP3, FLAC, and other formats (WAV works without) |
| libsndfile | Audio file I/O | Reading/writing audio files |
| PortAudio | Audio hardware access | Real-time recording (`bioamla realtime` commands) |

### Python Package

```bash
pip install bioamla
```

For GPU support, install PyTorch with CUDA first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bioamla
```

## Quick Start

**CLI:**

```bash
# Run inference on audio files
bioamla models predict ast ./audio/

# Initialize a project for organized workflows
bioamla project init -n "My Study"

# Generate spectrograms
bioamla audio visualize recording.wav -o spectrogram.png
```

**Python API:**

```python
from bioamla.models import load_model

model = load_model("ast")
result = model.predict_file("recording.wav")
print(f"{result.label}: {result.confidence:.2%}")
```

## Documentation

- **[CLI Getting Started Guide](docs/cli-start.md)** - Command-line usage and workflows
- **[API Getting Started Guide](docs/api-start.md)** - Using bioamla as a Python library

### Example Workflows

Bioamla includes ready-to-run shell scripts demonstrating various capabilities. Access them via the CLI:

```bash
bioamla examples list              # List all available examples
bioamla examples show 01           # Display example content
bioamla examples copy 01 ./        # Copy example to current directory
bioamla examples copy-all ./       # Copy all examples to a directory
bioamla examples info 01           # Show detailed information about an example
```

## Features

### Machine Learning & Inference

- Audio classification using Audio Spectrogram Transformer (AST) models
- Batch inference for large-scale analysis with CSV/JSON output
- Model fine-tuning on custom labeled datasets
- Embedding extraction for clustering and analysis
- Support for multiple backends: AST, BirdNET, OpenSoundscape, custom CNNs

### Audio Processing

- Signal filtering (bandpass, lowpass, highpass)
- Audio denoising (spectral and waveform methods)
- Silence-based segmentation and onset detection
- Normalization and resampling
- Data augmentation (noise injection, time stretch, pitch shift)

### Visualization & Analysis

- Spectrogram generation (mel, STFT, MFCC, waveform)
- Acoustic indices (ACI, ADI, AEI, BIO, NDSI)
- Dimensionality reduction (UMAP, t-SNE, PCA)
- Clustering (HDBSCAN, K-means, DBSCAN)
- Novelty detection for discovering unknown sounds

### Specialized Detection

- Band-limited energy detection
- Periodic call detection (e.g., frog calls)
- CWT peak detection
- Accelerating pattern detection

### Real-time & Active Learning

- Live audio recording with detection callbacks
- Real-time spectrogram streaming
- Active learning workflows with uncertainty sampling
- Annotation queue management

### Project Management

- Project-based workflows with TOML configuration
- Command history tracking
- Configuration hierarchy (project, user, system defaults)
- Multiple project templates (default, minimal, research, production)

### External Integrations

- iNaturalist audio search and download
- Xeno-canto recording search
- eBird observation integration
- PostgreSQL export

## CLI vs API

Bioamla can be used either as a command-line tool or as a Python library. Choose based on your workflow:

| Use Case                   | CLI                                                 | API                               |
| -------------------------- | --------------------------------------------------- | --------------------------------- |
| Quick one-off analysis     | `bioamla models predict ast ./audio/`               | -                                 |
| Batch processing scripts   | `bioamla models predict ast ./data/ -o results.csv` | -                                 |
| Integration in Python code | -                                                   | `model.predict_file("audio.wav")` |
| Custom pipelines           | -                                                   | Full access to all modules        |
| Interactive exploration    | -                                                   | Jupyter notebooks                 |
| Automation & cron jobs     | Shell scripts with CLI                              | Python scripts with API           |

**CLI Advantages:**

- No coding required
- Project system with config files
- Built-in command history
- Progress bars and formatted output

**API Advantages:**

- Full programmatic control
- Access to intermediate results (embeddings, spectrograms)
- Custom model pipelines
- Integration with other Python libraries

---

## API Reference

### Core Modules

#### Model Loading & Inference

```python
from bioamla.models import load_model
from bioamla.inference import ASTInference, run_batch_inference

# Load a model
model = load_model(model_type="ast", model_path="MIT/ast-finetuned-audioset-10-10-0.4593")

# Single file prediction
result = model.predict_file("audio.wav")
print(f"{result.label}: {result.confidence:.2%}")

# AST inference with segmentation
ast = ASTInference(model_path="MIT/ast-finetuned-audioset-10-10-0.4593")
results = ast.predict_segments("long_recording.wav", clip_length=10, overlap=2)

# Batch inference
from bioamla.inference import BatchInferenceConfig
config = BatchInferenceConfig(input_dir="./audio", output_file="results.csv")
run_batch_inference(config)

# Extract embeddings for clustering
embeddings = model.extract_embeddings("audio.wav", layer="last_hidden_state")
```

#### Audio Processing

```python
from bioamla.signal import (
    bandpass_filter, lowpass_filter, highpass_filter,
    denoise_spectrogram, denoise_waveform,
    segment_on_silence, detect_onsets,
    normalize_audio
)

# Filtering
filtered = bandpass_filter(audio, sample_rate=16000, low_freq=500, high_freq=8000)
filtered = lowpass_filter(audio, sample_rate=16000, cutoff_freq=4000)
filtered = highpass_filter(audio, sample_rate=16000, cutoff_freq=200)

# Denoising
clean_spec = denoise_spectrogram(spectrogram, method="spectral")
clean_audio = denoise_waveform(audio, sample_rate=16000)

# Segmentation
segments = segment_on_silence(audio, sample_rate=16000, threshold_db=-40)

# Event detection
onset_times = detect_onsets(audio, sample_rate=16000)

# Normalization
normalized = normalize_audio(audio, target_db=-20.0)
```

#### Audio Augmentation

```python
from bioamla.augment import AugmentationConfig, augment_file, batch_augment

config = AugmentationConfig(
    noise_snr=20,
    time_stretch_range=(0.9, 1.1),
    pitch_shift_range=(-2, 2),
    gain_db_range=(-6, 6)
)

# Single file
augment_file("input.wav", "output.wav", config)

# Batch augmentation
batch_augment("./raw_audio", "./augmented", config)
```

#### Visualization

```python
from bioamla.visualize import generate_spectrogram

# Generate spectrograms (mel, STFT, MFCC, or waveform)
generate_spectrogram(
    audio_path="audio.wav",
    output_path="spectrogram.png",
    viz_type="mel",      # "stft", "mel", "mfcc", "waveform"
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    cmap="magma"
)
```

#### Clustering & Discovery

```python
from bioamla.clustering import (
    AudioClusterer, reduce_dimensions,
    NoveltyDetector, discover_novel_sounds
)

# Dimensionality reduction
reduced = reduce_dimensions(embeddings, method="umap", n_components=2)  # or "tsne", "pca"

# Clustering
clusterer = AudioClusterer(method="hdbscan")  # or "kmeans", "dbscan"
labels = clusterer.fit_predict(embeddings)

# Novelty detection
detector = NoveltyDetector(method="isolation_forest")
novel_indices = detector.detect(embeddings, threshold=0.1)

# Discover unknown sounds
novel_samples = discover_novel_sounds(embeddings, known_labels, threshold=0.95)
```

#### Advanced ML

```python
from bioamla.ml import CNNClassifier, CRNNClassifier, Ensemble, HierarchicalClassifier

# Custom classifiers
cnn = CNNClassifier(n_classes=10, n_mels=128)
crnn = CRNNClassifier(n_classes=10, n_mels=128)

# Ensemble predictions
ensemble = Ensemble(models=[model1, model2, model3], strategy="average")  # or "voting", "weighted"
result = ensemble.predict("audio.wav")

# Hierarchical classification
hierarchy = {
    "bird": ["songbird", "raptor", "waterfowl"],
    "songbird": ["sparrow", "warbler", "finch"]
}
classifier = HierarchicalClassifier(hierarchy, base_model)
multilevel_result = classifier.predict("audio.wav")
```

#### Specialized Detection

```python
from bioamla.detection import (
    BandLimitedEnergyDetector,
    RibbitDetector,
    CWTPeakDetector,
    AcceleratingPatternDetector
)

# Energy-based detection in frequency band
detector = BandLimitedEnergyDetector(low_freq=500, high_freq=3000, threshold_db=-30)
detections = detector.detect(audio, sample_rate=16000)

# Periodic call detection (e.g., frog calls)
ribbit = RibbitDetector(pulse_rate_hz=10, tolerance=0.1)
pulses = ribbit.detect(audio, sample_rate=16000)

# Peak detection using continuous wavelet transform
peak_detector = CWTPeakDetector()
peaks = peak_detector.detect(audio, sample_rate=16000)

# Accelerating pattern detection
accel = AcceleratingPatternDetector(min_pulses=3, acceleration_threshold=0.1)
patterns = accel.detect(audio, sample_rate=16000)
```

#### Real-time Processing

```python
from bioamla.realtime import (
    LiveRecorder, RealtimeSpectrogram, ContinuousMonitor,
    list_audio_devices, get_default_input_device, test_recording
)

# List available audio devices
devices = list_audio_devices()
default_device = get_default_input_device()

# Test recording
audio = test_recording(duration=5.0, device=0)

# Live recording with detection
recorder = LiveRecorder(detector=model)
recorder.start()
# ... recording ...
events = recorder.get_events()
recorder.stop()

# Real-time spectrogram streaming
def on_spectrogram(spec):
    print(f"Received spectrogram: {spec.shape}")

stream = RealtimeSpectrogram(callback=on_spectrogram)
stream.start()
```

#### Active Learning

```python
from bioamla.active_learning import (
    ActiveLearner, Sample, AnnotationQueue,
    UncertaintySampler, DiversitySampler, QueryByCommitteeSampler
)

# Initialize active learner
sampler = UncertaintySampler(strategy="entropy")  # or "least_confidence", "margin"
learner = ActiveLearner(model=model, sampler=sampler)

# Query most informative samples
samples_to_annotate = learner.query(unlabeled_pool, n_samples=10)

# Teach with new annotation
learner.teach(sample, label="species_name")

# Retrain model with accumulated annotations
learner.retrain()
```

#### Acoustic Indices

```python
from bioamla.indices import (
    compute_aci, compute_adi, compute_aei,
    compute_bio, compute_ndsi, compute_all_indices
)

# Individual indices
aci = compute_aci(audio, sample_rate=16000)   # Acoustic Complexity Index
adi = compute_adi(audio, sample_rate=16000)   # Acoustic Diversity Index
aei = compute_aei(audio, sample_rate=16000)   # Acoustic Evenness Index
bio = compute_bio(audio, sample_rate=16000, min_freq=2000, max_freq=8000)  # Bioacoustic Index

# Normalized Difference Soundscape Index
ndsi = compute_ndsi(
    audio, sample_rate=16000,
    anthro_min=1000, anthro_max=2000,
    bio_min=2000, bio_max=8000
)

# Compute all indices at once
indices = compute_all_indices(audio, sample_rate=16000)
```

#### External Integrations

```python
from bioamla.integrations import EBirdClient, PostgreSQLExporter, match_detections_to_ebird
from bioamla.api import xeno_canto, macaulay, species

# eBird integration
ebird = EBirdClient(api_key="your_api_key")
observations = ebird.get_observations_nearby(lat=40.7128, lng=-74.0060, distance=10)
matches = match_detections_to_ebird(detections, observations)

# PostgreSQL export
exporter = PostgreSQLExporter(connection_string="postgresql://...")
exporter.export_detections("detections.csv", table_name="audio_detections")

# Xeno-canto API
recordings = xeno_canto.search(species="Turdus migratorius")

# Species name conversion
scientific = species.common_to_scientific("American Robin")
common = species.scientific_to_common("Turdus migratorius")
```

#### Dataset Utilities

```python
from bioamla.datasets import count_audio_files, validate_metadata
from bioamla.metadata import read_metadata_csv, write_metadata_csv
from bioamla.explore import scan_directory, get_label_summary, filter_audio_files

# Count and validate
count = count_audio_files("./audio")
is_valid = validate_metadata("./audio", "metadata.csv")

# Metadata operations
rows, fields = read_metadata_csv("metadata.csv")
write_metadata_csv("output.csv", rows, fields)

# Dataset exploration
files, info = scan_directory("./dataset", recursive=True)
summary = get_label_summary(files)
filtered = filter_audio_files(files, label="bird", split="train")
```

#### Configuration

```python
from bioamla.config import load_config, get_config, set_config, Config

# Load configuration (searches: CLI → project → cwd → home → system → defaults)
config = load_config()

# Access configuration values
sample_rate = config.audio.sample_rate
model_name = config.models.default_ast_model

# Modify configuration
set_config("inference.batch_size", 16)
```

#### Project Management

```python
from bioamla.project import create_project, load_project, find_project_root

# Create a new project
project = create_project(
    path="./my_study",
    name="Bird Survey 2025",
    template="research"  # "default", "minimal", "research", "production"
)

# Load existing project
project = load_project("./my_study")

# Find project root from subdirectory
root = find_project_root("./my_study/data/recordings")
```

#### Device Management

```python
from bioamla.device import get_device, get_device_info, move_to_device

# Get optimal device
device = get_device()  # Returns CUDA if available, else CPU

# Device information
info = get_device_info()
print(f"Device: {info['device']}, CUDA available: {info['cuda_available']}")

# Move tensors/models to device
model = move_to_device(model)
```

---

## CLI Commands

### Config Commands

```bash
bioamla config show                    # Display current configuration
bioamla config init                    # Create default config file
bioamla config path                    # Show config search paths
bioamla config purge --models          # Clear model cache
bioamla config purge --all             # Clear all caches
```

### Project Commands

```bash
bioamla project init -n "Study Name"   # Initialize new project
bioamla project init --template research
bioamla project status                 # Show project info
bioamla project config --show          # View project config
```

### Model Operations

```bash
# Prediction
bioamla models predict ast ./audio/
bioamla models predict ast --batch ./recordings/ --output results.csv

# Training
bioamla models train ast --data-dir ./training_data --epochs 10
bioamla models train cnn --data-dir ./data --classes 25

# Utilities
bioamla models list                    # Available models
bioamla models info --model-path ./model
bioamla models embed --input audio.wav --output embeddings.npy
bioamla models ensemble --model-dirs ./m1 ./m2 --strategy voting
```

### Audio Commands

```bash
# Information
bioamla audio list ./recordings/
bioamla audio info recording.wav

# Processing
bioamla audio filter --bandpass --low 500 --high 8000 ./audio/
bioamla audio denoise --method spectral ./audio/
bioamla audio segment --silence-threshold -40 ./audio/
bioamla audio normalize --target-db -20 ./audio/
bioamla audio resample --rate 16000 ./audio/

# Visualization
bioamla audio visualize --type mel --input audio.wav --output spec.png
```

### Detection

```bash
bioamla detect energy --low-freq 500 --high-freq 3000 ./audio/
bioamla detect ribbit --pulse-rate 10 ./audio/
bioamla detect peaks --snr 10 ./audio/
bioamla detect accelerating --min-pulses 3 ./audio/
```

### Indices Commands

```bash
bioamla indices compute ./audio/       # Compute all indices
bioamla indices aci ./audio/           # Acoustic Complexity Index
bioamla indices ndsi --anthro-min 1000 --anthro-max 2000 ./audio/
```

### Clustering

```bash
bioamla cluster reduce --embeddings emb.npy --method umap --output reduced.npy
bioamla cluster cluster --embeddings emb.npy --method hdbscan --output labels.npy
bioamla cluster novelty --embeddings emb.npy --output novel.npy
```

### Learn Commands

```bash
bioamla learn init --predictions predictions.csv
bioamla learn query --n-samples 20
bioamla learn annotate --annotations new_labels.csv
bioamla learn status
```

### Data Services

```bash
# iNaturalist
bioamla services inat search --species "Lithobates catesbeianus"
bioamla services inat download --species "Lithobates catesbeianus" --output-dir ./data

# Xeno-canto
bioamla services xc search --species "Turdus migratorius"
bioamla services xc download --species "Turdus migratorius"

# eBird
bioamla services ebird nearby --lat 40.7128 --lng -74.0060
bioamla services ebird validate --species-code "amerob"
```

### Real-time

```bash
bioamla realtime devices               # List audio input devices
bioamla realtime test --duration 5     # Test recording
```

### Examples

```bash
bioamla examples list                  # List available example workflows
bioamla examples show 01               # Display example content
bioamla examples copy 01 ./            # Copy to current directory
bioamla examples copy-all ./workflows  # Copy all examples
bioamla examples info 01               # Show example details
```

### System

```bash
bioamla devices                        # Show CUDA/GPU info
bioamla version                        # Show version
bioamla log show --limit 50            # Show command history
```

---

## Supported Model Backends

| Backend            | Description                   | Use Case                                  |
| ------------------ | ----------------------------- | ----------------------------------------- |
| **AST**            | Audio Spectrogram Transformer | General audio classification, fine-tuning |
| **BirdNET**        | Bird species classifier       | Bird identification                       |
| **OpenSoundscape** | ResNet-based CNNs             | Transfer learning                         |
| **Custom CNN**     | From-scratch training         | Domain-specific models                    |

---

## Related Projects

Bioamla builds on and complements other bioacoustics tools:

- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) - Bird sound identification
- [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) - Bioacoustic analysis toolkit
- [Koogu](https://github.com/shyamblast/Koogu) - Transfer learning for bioacoustics
- [audiomoth-utils](https://github.com/OpenAcousticDevices/AudioMoth-Utils) - AudioMoth recorder utilities

## Citation

```bibtex
@software{bioamla,
  author = {McMeen, John},
  title = {Bioamla: Bioacoustics and Machine Learning Applications},
  year = {2025},
  url = {https://github.com/jmcmeen/bioamla}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
