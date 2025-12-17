# Getting Started with Bioamla API

This guide covers using bioamla as a Python library for bioacoustic analysis and machine learning workflows.

## Installation

```bash
pip install bioamla
```

For GPU support, install PyTorch with CUDA first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bioamla
```

## Quick Start

```python
from bioamla.models import load_model

# Load a pre-trained model
model = load_model("ast")

# Classify an audio file
result = model.predict_file("recording.wav")
print(f"{result.label}: {result.confidence:.2%}")
```

## Model Loading & Inference

### Loading Models

```python
from bioamla.models import load_model

# Load default AST model
model = load_model("ast")

# Load specific model from Hugging Face
model = load_model(
    model_type="ast",
    model_path="MIT/ast-finetuned-audioset-10-10-0.4593"
)

# Load a fine-tuned local model
model = load_model(model_type="ast", model_path="./models/my_finetuned_model")
```

### Single File Prediction

```python
# Basic prediction
result = model.predict_file("audio.wav")
print(f"{result.label}: {result.confidence:.2%}")

# Get top-k predictions
results = model.predict_file("audio.wav", top_k=5)
for r in results:
    print(f"{r.label}: {r.confidence:.2%}")
```

### AST Inference with Segmentation

For long recordings, segment audio into clips:

```python
from bioamla.inference import ASTInference

ast = ASTInference(model_path="MIT/ast-finetuned-audioset-10-10-0.4593")

# Predict with segmentation
results = ast.predict_segments(
    "long_recording.wav",
    clip_length=10,    # seconds per clip
    overlap=2          # overlap between clips
)

for segment in results:
    print(f"Time {segment.start:.1f}-{segment.end:.1f}s: {segment.label}")
```

### Batch Inference

Process multiple files efficiently:

```python
from bioamla.inference import BatchInferenceConfig, run_batch_inference

config = BatchInferenceConfig(
    input_dir="./audio",
    output_file="results.csv",
    batch_size=8,
    recursive=True,
    top_k=5,
    min_confidence=0.01
)

results = run_batch_inference(config)
```

### Embedding Extraction

Extract feature vectors for downstream analysis:

```python
# Extract embeddings from audio
embeddings = model.extract_embeddings("audio.wav", layer="last_hidden_state")
print(f"Embedding shape: {embeddings.shape}")

# Batch embedding extraction
import numpy as np
from pathlib import Path

audio_files = list(Path("./audio").glob("*.wav"))
all_embeddings = []

for f in audio_files:
    emb = model.extract_embeddings(str(f))
    all_embeddings.append(emb)

embeddings_array = np.vstack(all_embeddings)
np.save("embeddings.npy", embeddings_array)
```

## Audio Processing

### Signal Filtering

```python
from bioamla.signal import bandpass_filter, lowpass_filter, highpass_filter
import torchaudio

# Load audio
waveform, sample_rate = torchaudio.load("audio.wav")
audio = waveform.numpy().squeeze()

# Apply filters
filtered = bandpass_filter(audio, sample_rate=sample_rate, low_freq=500, high_freq=8000)
filtered = lowpass_filter(audio, sample_rate=sample_rate, cutoff_freq=4000)
filtered = highpass_filter(audio, sample_rate=sample_rate, cutoff_freq=200)
```

### Denoising

```python
from bioamla.signal import denoise_spectrogram, denoise_waveform

# Denoise waveform directly
clean_audio = denoise_waveform(audio, sample_rate=16000)

# Denoise spectrogram
clean_spec = denoise_spectrogram(spectrogram, method="spectral")
```

### Segmentation

```python
from bioamla.signal import segment_on_silence, detect_onsets

# Segment audio based on silence
segments = segment_on_silence(audio, sample_rate=16000, threshold_db=-40)
for start, end in segments:
    print(f"Segment: {start:.2f}s - {end:.2f}s")

# Detect sound onsets
onset_times = detect_onsets(audio, sample_rate=16000)
```

### Normalization

```python
from bioamla.signal import normalize_audio

# Normalize to target dB level
normalized = normalize_audio(audio, target_db=-20.0)
```

## Data Augmentation

```python
from bioamla.augment import AugmentationConfig, augment_file, batch_augment

config = AugmentationConfig(
    noise_snr=20,                    # Add noise at 20dB SNR
    time_stretch_range=(0.9, 1.1),   # Time stretch 90-110%
    pitch_shift_range=(-2, 2),       # Pitch shift +/- 2 semitones
    gain_db_range=(-6, 6)            # Volume variation
)

# Augment single file
augment_file("input.wav", "output.wav", config)

# Batch augmentation
batch_augment("./raw_audio", "./augmented", config)
```

## Visualization

### Generate Spectrograms

```python
from bioamla.visualize import generate_spectrogram

# Mel spectrogram
generate_spectrogram(
    audio_path="audio.wav",
    output_path="mel_spec.png",
    viz_type="mel",
    n_mels=128,
    n_fft=2048,
    hop_length=512,
    cmap="magma"
)

# Other types: "stft", "mfcc", "waveform"
generate_spectrogram("audio.wav", "stft_spec.png", viz_type="stft")
generate_spectrogram("audio.wav", "mfcc_spec.png", viz_type="mfcc")
generate_spectrogram("audio.wav", "waveform.png", viz_type="waveform")
```

## Clustering & Dimensionality Reduction

### Reduce Dimensions

```python
from bioamla.clustering import reduce_dimensions
import numpy as np

# Load embeddings
embeddings = np.load("embeddings.npy")

# UMAP reduction
reduced_umap = reduce_dimensions(embeddings, method="umap", n_components=2)

# t-SNE reduction
reduced_tsne = reduce_dimensions(embeddings, method="tsne", n_components=2)

# PCA reduction
reduced_pca = reduce_dimensions(embeddings, method="pca", n_components=2)
```

### Clustering

```python
from bioamla.clustering import AudioClusterer

# HDBSCAN clustering
clusterer = AudioClusterer(method="hdbscan")
labels = clusterer.fit_predict(embeddings)

# K-means clustering
clusterer = AudioClusterer(method="kmeans", n_clusters=10)
labels = clusterer.fit_predict(embeddings)

# DBSCAN clustering
clusterer = AudioClusterer(method="dbscan", eps=0.5, min_samples=5)
labels = clusterer.fit_predict(embeddings)
```

### Novelty Detection

```python
from bioamla.clustering import NoveltyDetector, discover_novel_sounds

# Detect outliers with isolation forest
detector = NoveltyDetector(method="isolation_forest")
novel_indices = detector.detect(embeddings, threshold=0.1)
print(f"Found {len(novel_indices)} novel samples")

# Discover sounds not matching known labels
novel_samples = discover_novel_sounds(embeddings, known_labels, threshold=0.95)
```

## Specialized Detection

### Energy-Based Detection

```python
from bioamla.detection import BandLimitedEnergyDetector

detector = BandLimitedEnergyDetector(
    low_freq=500,
    high_freq=3000,
    threshold_db=-30
)

detections = detector.detect(audio, sample_rate=16000)
for det in detections:
    print(f"Detection at {det.start:.2f}s - {det.end:.2f}s")
```

### Periodic Call Detection

```python
from bioamla.detection import RibbitDetector

# Detect frog-like periodic calls
ribbit = RibbitDetector(pulse_rate_hz=10, tolerance=0.1)
pulses = ribbit.detect(audio, sample_rate=16000)
```

### Pattern Detection

```python
from bioamla.detection import CWTPeakDetector, AcceleratingPatternDetector

# CWT peak detection
peak_detector = CWTPeakDetector()
peaks = peak_detector.detect(audio, sample_rate=16000)

# Accelerating pattern detection
accel = AcceleratingPatternDetector(min_pulses=3, acceleration_threshold=0.1)
patterns = accel.detect(audio, sample_rate=16000)
```

## Acoustic Indices

```python
from bioamla.indices import (
    compute_aci, compute_adi, compute_aei,
    compute_bio, compute_ndsi, compute_all_indices
)

# Individual indices
aci = compute_aci(audio, sample_rate=16000)   # Acoustic Complexity Index
adi = compute_adi(audio, sample_rate=16000)   # Acoustic Diversity Index
aei = compute_aei(audio, sample_rate=16000)   # Acoustic Evenness Index

# Bioacoustic Index (for specific frequency range)
bio = compute_bio(audio, sample_rate=16000, min_freq=2000, max_freq=8000)

# Normalized Difference Soundscape Index
ndsi = compute_ndsi(
    audio,
    sample_rate=16000,
    anthro_min=1000,   # Anthropogenic noise band
    anthro_max=2000,
    bio_min=2000,      # Biophony band
    bio_max=8000
)

# Compute all indices at once
indices = compute_all_indices(audio, sample_rate=16000)
print(f"ACI: {indices['aci']:.3f}, NDSI: {indices['ndsi']:.3f}")
```

## Advanced ML

### Custom Classifiers

```python
from bioamla.ml import CNNClassifier, CRNNClassifier

# CNN classifier
cnn = CNNClassifier(n_classes=10, n_mels=128)

# CRNN (CNN + RNN) classifier
crnn = CRNNClassifier(n_classes=10, n_mels=128)
```

### Ensemble Models

```python
from bioamla.ml import Ensemble

# Create ensemble from multiple models
ensemble = Ensemble(
    models=[model1, model2, model3],
    strategy="average"  # or "voting", "weighted"
)

result = ensemble.predict("audio.wav")
```

### Hierarchical Classification

```python
from bioamla.ml import HierarchicalClassifier

# Define taxonomy
hierarchy = {
    "bird": ["songbird", "raptor", "waterfowl"],
    "songbird": ["sparrow", "warbler", "finch"]
}

classifier = HierarchicalClassifier(hierarchy, base_model)
result = classifier.predict("audio.wav")

# Access predictions at each level
print(f"Top level: {result.top_level}")
print(f"Species: {result.species}")
```

## Real-time Processing

### Audio Device Management

```python
from bioamla.realtime import list_audio_devices, get_default_input_device, test_recording

# List available audio devices
devices = list_audio_devices()
for dev in devices:
    print(f"{dev['index']}: {dev['name']}")

# Get default input device
default_device = get_default_input_device()

# Test recording
audio = test_recording(duration=5.0, device=0)
```

### Live Recording with Detection

```python
from bioamla.realtime import LiveRecorder

recorder = LiveRecorder(detector=model)
recorder.start()

# Recording in progress...
import time
time.sleep(30)

# Get detected events
events = recorder.get_events()
recorder.stop()

for event in events:
    print(f"Detected {event.label} at {event.time:.2f}s")
```

### Real-time Spectrogram Streaming

```python
from bioamla.realtime import RealtimeSpectrogram

def on_spectrogram(spec):
    print(f"Received spectrogram: {spec.shape}")
    # Process or display spectrogram

stream = RealtimeSpectrogram(callback=on_spectrogram)
stream.start()

# ... streaming ...

stream.stop()
```

## Active Learning

```python
from bioamla.active_learning import (
    ActiveLearner, Sample, AnnotationQueue,
    UncertaintySampler, DiversitySampler
)

# Initialize with uncertainty sampling
sampler = UncertaintySampler(strategy="entropy")  # or "least_confidence", "margin"
learner = ActiveLearner(model=model, sampler=sampler)

# Query most informative samples from unlabeled pool
samples_to_annotate = learner.query(unlabeled_pool, n_samples=10)

# After annotation, teach the model
for sample in annotated_samples:
    learner.teach(sample, label=sample.annotation)

# Retrain with accumulated annotations
learner.retrain()
```

## External Integrations

### Xeno-canto API

```python
from bioamla.api import xeno_canto

# Search for recordings
recordings = xeno_canto.search(species="Turdus migratorius")

for rec in recordings[:5]:
    print(f"{rec.species}: {rec.url}")
```

### Species Name Conversion

```python
from bioamla.api import species

# Convert between common and scientific names
scientific = species.common_to_scientific("American Robin")
print(scientific)  # "Turdus migratorius"

common = species.scientific_to_common("Turdus migratorius")
print(common)  # "American Robin"
```

### eBird Integration

```python
from bioamla.integrations import EBirdClient, match_detections_to_ebird

# Initialize client (requires API key)
ebird = EBirdClient(api_key="your_api_key")

# Get nearby observations
observations = ebird.get_observations_nearby(
    lat=40.7128,
    lng=-74.0060,
    distance=10  # km
)

# Match your detections to eBird observations
matches = match_detections_to_ebird(detections, observations)
```

### Database Export

```python
from bioamla.integrations import PostgreSQLExporter

exporter = PostgreSQLExporter(connection_string="postgresql://user:pass@host/db")
exporter.export_detections("detections.csv", table_name="audio_detections")
```

## Configuration

### Loading Configuration

```python
from bioamla.config import load_config, get_config, set_config

# Load configuration (searches multiple locations)
config = load_config()

# Access values
sample_rate = config.audio.sample_rate
model_name = config.models.default_model
batch_size = config.inference.batch_size

# Modify runtime configuration
set_config("inference.batch_size", 16)
```

### Configuration Search Order

Configuration files are loaded in this priority order:

1. Explicit path passed to `load_config(path)`
2. Project config (`.bioamla/config.toml`)
3. Current directory (`./bioamla.toml`)
4. User config (`~/.config/bioamla/config.toml`)
5. System config (`/etc/bioamla/config.toml`)
6. Built-in defaults

## Project Management

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

## Device Management

```python
from bioamla.device import get_device, get_device_info, move_to_device

# Get optimal device (CUDA if available)
device = get_device()
print(f"Using device: {device}")

# Get detailed device info
info = get_device_info()
print(f"CUDA available: {info['cuda_available']}")
if info['cuda_available']:
    print(f"GPU: {info['gpu_name']}")

# Move model/tensor to device
model = move_to_device(model)
```

## Dataset Utilities

```python
from bioamla.datasets import count_audio_files, validate_metadata
from bioamla.metadata import read_metadata_csv, write_metadata_csv
from bioamla.explore import scan_directory, get_label_summary, filter_audio_files

# Count audio files
count = count_audio_files("./audio")
print(f"Found {count} audio files")

# Validate metadata
is_valid = validate_metadata("./audio", "metadata.csv")

# Read/write metadata
rows, fields = read_metadata_csv("metadata.csv")
write_metadata_csv("output.csv", rows, fields)

# Explore dataset
files, info = scan_directory("./dataset", recursive=True)
summary = get_label_summary(files)
print(f"Labels: {summary}")

# Filter files
filtered = filter_audio_files(files, label="bird", split="train")
```

## Example Workflows

### Species Classification Pipeline

```python
from bioamla.models import load_model
from bioamla.inference import BatchInferenceConfig, run_batch_inference
from bioamla.visualize import generate_spectrogram
from pathlib import Path

# Load model
model = load_model("ast")

# Process directory
config = BatchInferenceConfig(
    input_dir="./recordings",
    output_file="results.csv",
    top_k=5
)
results = run_batch_inference(config)

# Generate spectrograms for high-confidence detections
for result in results:
    if result.confidence > 0.8:
        generate_spectrogram(
            result.file_path,
            f"./spectrograms/{Path(result.file_path).stem}.png"
        )
```

### Soundscape Analysis

```python
from bioamla.indices import compute_all_indices
from pathlib import Path
import torchaudio
import pandas as pd

audio_dir = Path("./soundscape_recordings")
results = []

for audio_file in audio_dir.glob("*.wav"):
    waveform, sr = torchaudio.load(audio_file)
    audio = waveform.numpy().squeeze()

    indices = compute_all_indices(audio, sample_rate=sr)
    indices['file'] = audio_file.name
    results.append(indices)

df = pd.DataFrame(results)
df.to_csv("soundscape_indices.csv", index=False)
```

### Embedding-Based Clustering

```python
from bioamla.models import load_model
from bioamla.clustering import reduce_dimensions, AudioClusterer
import numpy as np
from pathlib import Path

model = load_model("ast")
audio_files = list(Path("./audio").glob("*.wav"))

# Extract embeddings
embeddings = []
for f in audio_files:
    emb = model.extract_embeddings(str(f))
    embeddings.append(emb)

embeddings = np.vstack(embeddings)

# Reduce dimensions
reduced = reduce_dimensions(embeddings, method="umap", n_components=2)

# Cluster
clusterer = AudioClusterer(method="hdbscan")
labels = clusterer.fit_predict(embeddings)

# Save results
np.save("embeddings.npy", embeddings)
np.save("reduced.npy", reduced)
np.save("labels.npy", labels)
```

### Active Learning Loop

```python
from bioamla.models import load_model
from bioamla.active_learning import ActiveLearner, UncertaintySampler

model = load_model("ast")
sampler = UncertaintySampler(strategy="entropy")
learner = ActiveLearner(model=model, sampler=sampler)

# Initial unlabeled pool
unlabeled_files = list(Path("./unlabeled").glob("*.wav"))

# Active learning iterations
for iteration in range(5):
    # Query uncertain samples
    to_annotate = learner.query(unlabeled_files, n_samples=20)

    print(f"Iteration {iteration + 1}: Review these files:")
    for sample in to_annotate:
        print(f"  - {sample.path}")

    # (User annotates samples externally)
    # annotations = load_annotations("annotations.csv")

    # Teach model with new annotations
    # for sample, label in annotations:
    #     learner.teach(sample, label)

    # Retrain
    # learner.retrain()
```
