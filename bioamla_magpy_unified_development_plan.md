# Bioamla + MagPy Unified Development Plan

**Development Philosophy:** Prototype rapidly in `jmcmeen/magpy` (GUI), then extract and harden libraries into `jmcmeen/bioamla` (CLI engine/API).

---

## Executive Summary

This plan consolidates seven planning documents into a unified roadmap for building a comprehensive bioacoustics analysis platform. The architecture separates concerns cleanly:

| Repository        | Role                                             | Primary Users                   |
| ----------------- | ------------------------------------------------ | ------------------------------- |
| `jmcmeen/bioamla` | Core API engine, CLI, database, ML services      | Developers, scripts, automation |
| `jmcmeen/magpy`   | PyQt6 GUI, prototyping sandbox, visual workflows | Researchers, end users          |

**Package Structure:**
- `bioamla/core/` - Base functionality and API layer (audio, ml, analysis, detection, services, files, workflow)
- `bioamla/commands/` - Command pattern infrastructure for external app integration
- `bioamla/database/` - Persistence layer (SQLModel, repositories, Unit of Work)
- `bioamla/views/` - Interface layer (CLI via Click)

**Development flow:** Features are prototyped in MagPy's interactive environment, then the underlying logic is extracted, tested, and moved to bioamla as stable library code. MagPy then imports from bioamla, completing the cycle.

---

## Part 1: Architecture Overview

### 1.1 Hexagonal Architecture

Both projects follow hexagonal (ports-and-adapters) architecture, enabling clean separation and multiple interface types.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BIOAMLA CORE                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Domain Layer                                     ││
│  │  • Audio, Spectrogram, Annotation entities                               ││
│  │  • Detection, Classification, Analysis services                          ││
│  │  • Workflow orchestration logic                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐│
│  │  Repository Layer │  │   Service Layer   │  │    External Adapters      ││
│  │  • SQLModel + UoW │  │  • CommandService │  │  • iNaturalist, eBird     ││
│  │  • Generic repos  │  │  • InferenceService│  │  • Xeno-canto, HuggingFace││
│  │  • Specifications │  │  • WorkflowService │  │  • BirdNET, OpenSoundscape││
│  └───────────────────┘  └───────────────────┘  └───────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                          Interface Ports                                 ││
│  │  CLI (Click) │ TUI (Textual) │ API (FastAPI) │ GUI (via MagPy)          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               MAGPY GUI                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      Multi-Screen Workspaces                             ││
│  │  1. Spectral Analyzer    2. Dataset Browser    3. Terminal               ││
│  │  4. Pipeline Editor      5. Command Builder    6. Services Browser       ││
│  │  7. LLM Assistant                                                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐│
│  │   Qt Widgets      │  │   Controllers     │  │     Workers (QThread)     ││
│  │  • Waveform       │  │  • AudioController│  │  • DetectionWorker        ││
│  │  • Spectrogram    │  │  • ProjectController│ │  • TrainingWorker         ││
│  │  • AnnotationTable│  │  • WorkflowController││  • IndexingWorker         ││
│  └───────────────────┘  └───────────────────┘  └───────────────────────────┘│
│                              ↓ imports ↓                                     │
│                         bioamla (core engine)                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Layer Architecture (SQLModel + Repository + Unit of Work)

The data layer uses patterns from the SQLModel guide for robust data management:

```python
# bioamla/database/unit_of_work.py
class UnitOfWork:
    """Coordinates repository operations within a transaction."""

    def __init__(self, session_factory):
        self._session_factory = session_factory

    def __enter__(self):
        self._session = self._session_factory()
        self.projects = ProjectRepository(self._session)
        self.recordings = RecordingRepository(self._session)
        self.annotations = AnnotationRepository(self._session)
        self.detections = DetectionRepository(self._session)
        self.workflows = WorkflowRepository(self._session)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        self.close()

    def commit(self): self._session.commit()
    def rollback(self): self._session.rollback()
    def close(self): self._session.close()
```

### 1.3 Command Pattern for Operations

All operations implement the Command pattern for undo/redo support (from audio editor plan):

```python
# bioamla/commands/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class CommandResult:
    success: bool
    message: str
    data: Any = None

class Command(ABC):
    """Base class for all undoable operations."""

    description: str = "Unknown operation"

    @abstractmethod
    def execute(self) -> CommandResult:
        """Execute the command."""
        pass

    @abstractmethod
    def undo(self) -> CommandResult:
        """Reverse the command."""
        pass

    def redo(self) -> CommandResult:
        """Re-execute (default: call execute again)."""
        return self.execute()
```

---

## Part 2: Feature Gap Analysis & Prioritization

### 2.1 Current State vs. Target State

| Feature                       | Current bioamla | Target                                 | Priority |
| ----------------------------- | --------------- | -------------------------------------- | -------- |
| **Audio Processing**          | ✅ Full          | —                                      | —        |
| **AST Inference/Training**    | ✅ Full          | —                                      | —        |
| **Data Augmentation**         | ✅ Full          | —                                      | —        |
| **iNaturalist Integration**   | ✅ Full          | —                                      | —        |
| **Project Repository System** | ❌ None          | `.bioamla/` directories                | P0       |
| **Acoustic Indices**          | ❌ None          | ACI, ADI, NDSI, BIO, H via scikit-maad | P1       |
| **RIBBIT Pulse Detection**    | ❌ None          | OpenSoundscape integration             | P1       |
| **Embedding Extraction**      | ❌ None          | `bioamla ast embed` command            | P1       |
| **UMAP/HDBSCAN Clustering**   | ❌ None          | Vocal repertoire discovery             | P2       |
| **Xeno-canto Integration**    | ❌ None          | xenopy wrapper                         | P2       |
| **LLM Command Generation**    | ❌ None          | Qwen2.5-Coder + RAG                    | P2       |
| **Workflow Pipeline System**  | ❌ None          | TOML + Jinja2 workflows                | P2       |
| **Generic Repository Layer**  | ❌ None          | SQLModel + UoW patterns                | P1       |

### 2.2 Reference Application Feature Parity

Target feature parity with leading bioacoustics tools:

| Feature                    | Raven   | Kaleidoscope | OpenSoundscape | bioamla Target     |
| -------------------------- | ------- | ------------ | -------------- | ------------------ |
| Spectrogram display        | ✅       | ✅            | ✅              | MagPy Phase 1      |
| Time-frequency annotations | ✅       | ❌            | ✅              | MagPy Phase 2      |
| 70+ acoustic measurements  | ✅       | Limited      | Limited        | bioamla P1         |
| Acoustic indices (25)      | ❌       | ✅            | ❌              | bioamla P1         |
| CNN training pipeline      | ❌       | ❌            | ✅              | Already done (AST) |
| Automated clustering       | ❌       | ✅            | Via sklearn    | bioamla P2         |
| Batch processing           | Limited | ✅            | ✅              | Already done       |
| Python scripting           | Via R   | ❌            | ✅              | Native             |

---

## Part 3: Development Phases

### Phase 0: Foundation Infrastructure (Weeks 1-4)

**Goal:** Establish project system and data layer before feature work.

#### 0.1 Project Repository System (bioamla)

Implement the `.bioamla/` directory convention from the project system plan:

```
my-bioacoustics-study/
├── .bioamla/
│   ├── config.toml          # Project configuration
│   ├── models.toml          # Model registry
│   ├── workflows/           # Saved workflow definitions
│   └── logs/
│       └── command_history.jsonl
├── audio/
├── outputs/
└── data/
```

**Deliverables:**

- [ ] `bioamla/core/project.py` - Project discovery, creation, management
- [ ] `bioamla/core/config.py` - Extended with cascade loading (project → user → system → defaults)
- [ ] `bioamla/core/command_log.py` - JSON Lines command history
- [ ] CLI commands: `bioamla project init|status|config`
- [ ] CLI commands: `bioamla log show|search|clear`
- [ ] Config templates: `default.toml`, `research.toml`, `production.toml`

#### 0.2 Database Layer (bioamla)

Implement SQLModel with Generic Repository and Unit of Work:

**Deliverables:**

- [ ] `bioamla/database/connection.py` - Engine and session factory
- [ ] `bioamla/database/models.py` - SQLModel entities (Project, Recording, Annotation, Detection)
- [ ] `bioamla/database/repository.py` - Generic BaseRepository
- [ ] `bioamla/database/unit_of_work.py` - UoW implementation
- [ ] `bioamla/database/repositories/` - Concrete repositories

#### 0.3 Command Pattern Infrastructure (bioamla)

**Deliverables:**

- [ ] `bioamla/commands/base.py` - Command ABC and UndoManager
- [ ] `bioamla/commands/audio.py` - Audio operation commands
- [ ] `bioamla/core/base_api.py` - `@config_aware` decorator for API methods

---

### Phase 1: Core Analysis Expansion (Weeks 5-12)

**Goal:** Fill critical feature gaps in bioamla's analysis capabilities.

#### 1.1 Acoustic Indices Module (bioamla)

Wrap scikit-maad for soundscape ecology indices:

```python
# bioamla/core/analysis/indices.py
from maad import sound, features

def calculate_indices(audio_path: Path, indices: list[str] = None) -> dict:
    """Calculate acoustic indices for an audio file.

    Available indices: ACI, ADI, AEI, BIO, NDSI, H, M, ...
    """
    s, fs = sound.load(str(audio_path))
    results = {}

    if "ACI" in indices:
        results["ACI"] = features.acoustic_complexity_index(s, fs)
    if "ADI" in indices:
        results["ADI"] = features.acoustic_diversity_index(s, fs)
    if "NDSI" in indices:
        results["NDSI"] = features.normalized_difference_soundscape_index(s, fs)
    # ... additional indices

    return results
```

**CLI:**

```bash
bioamla analysis indices ./recordings --batch \
  --indices ACI,ADI,NDSI,BIO \
  --output indices.csv \
  --temporal 1-minute  # Calculate per minute
```

**Deliverables:**

- [ ] `bioamla/core/analysis/indices.py` - Wrapper around scikit-maad
- [ ] CLI command: `bioamla analysis indices`
- [ ] Integration with batch processing pipeline
- [ ] Temporal resolution options (per-file, per-minute, per-hour)

#### 1.2 Embedding Extraction (bioamla)

Add dedicated embedding command for clustering workflows:

```python
# bioamla/core/ml/embeddings.py
def extract_embeddings(
    audio_paths: list[Path],
    model_path: str,
    layer: str = "penultimate",
    batch_size: int = 16
) -> np.ndarray:
    """Extract embeddings from audio files using AST model."""
    model = AutoModel.from_pretrained(model_path)
    # Extract from specified layer
    ...
```

**CLI:**

```bash
bioamla ast embed ./audio --batch \
  --model-path MIT/ast-finetuned-audioset-10-10-0.4593 \
  --output embeddings.npy \
  --layer penultimate \
  --normalize
```

**Deliverables:**

- [ ] `bioamla/core/ml/embeddings.py` - Embedding extraction
- [ ] CLI command: `bioamla ast embed`
- [ ] Support for multiple output formats (npy, parquet, csv)
- [ ] Optional PCA/UMAP dimensionality reduction

#### 1.3 Clustering & Novelty Detection (bioamla)

Implement unsupervised discovery pipeline:

```python
# bioamla/core/analysis/clustering.py
import umap
import hdbscan

def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    n_components: int = 8,
    min_cluster_size: int = 10
) -> ClusteringResult:
    """Cluster embeddings for vocal repertoire discovery."""
    reducer = umap.UMAP(n_components=n_components)
    reduced = reducer.fit_transform(embeddings)

    if method == "hdbscan":
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)
    # ... additional methods

    return ClusteringResult(labels=labels, reduced=reduced)
```

**CLI:**

```bash
bioamla analysis cluster embeddings.npy \
  --method hdbscan \
  --min-cluster-size 10 \
  --output clusters.csv \
  --plot clusters.png
```

**Deliverables:**

- [ ] `bioamla/core/analysis/clustering.py` - UMAP + HDBSCAN pipeline
- [ ] CLI command: `bioamla analysis cluster`
- [ ] Visualization output (scatter plot with cluster colors)
- [ ] Cluster export with file mappings

#### 1.4 RIBBIT Integration (bioamla)

Wrap OpenSoundscape RIBBIT for periodic call detection:

```python
# bioamla/core/detection/ribbit.py
from opensoundscape import ribbit

def detect_periodic_calls(
    audio_path: Path,
    signal_band: tuple[float, float],
    noise_bands: list[tuple[float, float]],
    pulse_rate_range: tuple[float, float]
) -> pd.DataFrame:
    """Detect periodic vocalizations using RIBBIT algorithm."""
    ...
```

**Deliverables:**

- [ ] `bioamla/core/detection/ribbit.py` - OpenSoundscape wrapper
- [ ] CLI command: `bioamla detect ribbit`
- [ ] Species profile presets for common amphibians

#### 1.5 Xeno-canto Integration (bioamla)

Add Xeno-canto data source alongside iNaturalist:

```bash
# CLI commands
bioamla xc search --species "Phylloscopus collybita" --quality A --length 5-30
bioamla xc download ./search_results.json --output ./xc_audio --format mp3
bioamla xc metadata XC123456  # Get recording metadata
```

**Deliverables:**

- [ ] `bioamla/core/services/xeno_canto.py` - xenopy wrapper
- [ ] CLI command group: `bioamla xc`
- [ ] Rate limiting and caching
- [ ] Integration with dataset pipeline

---

### Phase 2: MagPy Core Viewer (Weeks 13-20)

**Goal:** Build foundational GUI with audio visualization and playback.

#### 2.1 Application Shell (MagPy)

```
magpy/
├── __init__.py
├── __main__.py              # Entry point
├── app.py                   # QApplication setup
├── main_window.py           # Multi-screen workspace manager
├── screens/
│   ├── spectral_analyzer.py # Screen 1: Audio viewer
│   ├── dataset_browser.py   # Screen 2: File management
│   ├── terminal.py          # Screen 3: Embedded CLI
│   ├── pipeline_editor.py   # Screen 4: Visual workflows
│   ├── command_builder.py   # Screen 5: GUI command construction
│   ├── services_browser.py  # Screen 6: API integrations
│   └── llm_assistant.py     # Screen 7: Natural language interface
├── widgets/
│   ├── waveform.py          # PyQtGraph waveform
│   ├── spectrogram.py       # PyQtGraph spectrogram
│   ├── annotation_table.py  # QTableView for selections
│   └── transport.py         # Playback controls
├── controllers/
│   └── audio_controller.py  # Wraps bioamla.audio
└── workers/
    └── detection_worker.py  # QThread for ML inference
```

#### 2.2 Spectral Analyzer Screen (MagPy)

**Deliverables:**

- [ ] Waveform display with zoom/pan (PyQtGraph)
- [ ] Spectrogram display with configurable FFT parameters
- [ ] Synchronized playhead across views
- [ ] Transport controls (play, pause, stop, loop)
- [ ] Selection tool for time-frequency regions
- [ ] Properties panel (duration, sample rate, channels)

#### 2.3 Audio Controller Integration

```python
# magpy/controllers/audio_controller.py
from bioamla.core.audio import Audio
from bioamla.core.spectrogram import Spectrogram

class AudioController:
    """Bridge between MagPy GUI and bioamla core."""

    def __init__(self):
        self._audio: Audio | None = None
        self._spectrogram: Spectrogram | None = None
        self._undo_manager = UndoManager(max_levels=100)

    def load_file(self, path: str) -> None:
        self._audio = Audio.from_file(path)
        self._spectrogram = Spectrogram.from_audio(self._audio)

    def apply_command(self, command: Command) -> CommandResult:
        """Execute command and add to undo stack."""
        result = command.execute()
        if result.success:
            self._undo_manager.push(command)
        return result
```

#### 2.4 Terminal Screen (MagPy)

Embedded terminal for direct bioamla CLI access:

**Deliverables:**

- [ ] QPlainTextEdit-based terminal emulator
- [ ] Command history (up/down arrows)
- [ ] Auto-completion for bioamla commands
- [ ] Real-time output streaming
- [ ] `undo`, `redo`, `history` commands

---

### Phase 3: MagPy Annotation & Detection (Weeks 21-28)

**Goal:** Add annotation creation and ML detection visualization.

#### 3.1 Annotation Interface

**Deliverables:**

- [ ] Click-drag selection box drawing on spectrogram
- [ ] Selection table panel (time start, end, freq low, high, label)
- [ ] Raven selection table import/export
- [ ] Keyboard shortcuts for rapid annotation (1-9 for labels)
- [ ] Undo/redo for annotation operations
- [ ] Export to CSV, Parquet, JSON

#### 3.2 Detection Interface

**Deliverables:**

- [ ] Model selection dialog (AST, BirdNET, custom)
- [ ] Detection progress with cancel support (QThread worker)
- [ ] Detection overlay on spectrogram (colored boxes)
- [ ] Results table with sorting, filtering, confidence threshold
- [ ] Detection → annotation conversion
- [ ] Batch detection dialog

---

### Phase 4: Workflow Pipeline System (Weeks 29-36)

**Goal:** Implement TOML-based workflow definitions with visual editor.

#### 4.1 Workflow Engine (bioamla)

```toml
# .bioamla/workflows/bird_detection.toml
[workflow]
name = "Bird Detection Pipeline"
version = "1.0"
description = "Standard workflow for field recordings"

[variables]
input_dir = { type = "path", required = true }
output_dir = { type = "path", default = "./outputs" }
model = { type = "string", default = "bioamla/bird-classifier" }
confidence_threshold = { type = "float", default = 0.7 }

[[steps]]
name = "preprocess"
command = "bioamla audio filter"
args = { input = "{{ input_dir }}", bandpass = "500-10000", output = "{{ output_dir }}/filtered" }

[[steps]]
name = "normalize"
command = "bioamla audio normalize"
args = { input = "{{ steps.preprocess.output }}", target_db = -20, output = "{{ output_dir }}/normalized" }
depends_on = ["preprocess"]

[[steps]]
name = "detect"
command = "bioamla ast predict"
args = { input = "{{ steps.normalize.output }}", model_path = "{{ model }}", output = "{{ output_dir }}/detections.csv" }
depends_on = ["normalize"]
```

**CLI:**

```bash
bioamla workflow run bird_detection.toml --input-dir ./recordings
bioamla workflow export bird_detection.toml --format bash > run_detection.sh
bioamla workflow validate bird_detection.toml
```

**Deliverables:**

- [ ] `bioamla/core/workflow/parser.py` - TOML workflow parser
- [ ] `bioamla/core/workflow/engine.py` - Execution engine with Jinja2 templating
- [ ] `bioamla/core/workflow/validator.py` - Schema validation
- [ ] CLI commands: `bioamla workflow run|export|validate|list`
- [ ] Bidirectional TOML ↔ shell script conversion

#### 4.2 Visual Pipeline Editor (MagPy)

Node-based workflow editor:

**Deliverables:**

- [ ] Node palette (drag bioamla commands onto canvas)
- [ ] Visual connection of step outputs to inputs
- [ ] Parameter editing panel per node
- [ ] TOML import/export
- [ ] Run workflow with progress visualization

---

### Phase 5: LLM Integration (Weeks 37-44)

**Goal:** Natural language → bioamla command generation.

#### 5.1 RAG Infrastructure (bioamla)

```python
# bioamla/core/llm/rag.py
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

class DocumentationRAG:
    """RAG system for bioamla documentation retrieval."""

    def __init__(self, db_path: Path = Path.home() / ".bioamla" / "rag"):
        self.client = PersistentClient(path=str(db_path))
        self.embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
        self.collection = self.client.get_or_create_collection("bioamla_docs")

    def index_documentation(self, docs_path: Path) -> None:
        """Index bioamla documentation and examples."""
        ...

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve relevant documentation for a query."""
        ...
```

#### 5.2 Command Generation Service (bioamla)

```python
# bioamla/core/llm/generator.py
class CommandGenerationService:
    """Generate bioamla commands from natural language."""

    def __init__(self, model_path: str, rag: DocumentationRAG):
        self.llm = Llama(model_path=model_path)
        self.rag = rag

    def generate(self, user_request: str) -> GeneratedCommand:
        """Generate a bioamla command from natural language."""
        context = self.rag.retrieve(user_request)
        prompt = self._build_prompt(user_request, context)
        response = self.llm(prompt, grammar=self.bioamla_grammar)
        return self._parse_response(response)

    def validate_command(self, command: str) -> ValidationResult:
        """Validate generated command before execution."""
        ...
```

#### 5.3 LLM Assistant Screen (MagPy)

**Deliverables:**

- [ ] Chat interface for natural language input
- [ ] Generated command preview with syntax highlighting
- [ ] One-click execution with confirmation
- [ ] Command history and regeneration
- [ ] Inline documentation lookup

---

### Phase 6: Advanced Features (Weeks 45-52)

#### 6.1 Training Wizard (MagPy)

Visual interface for model fine-tuning:

- [ ] Dataset selection and splitting
- [ ] Hyperparameter configuration
- [ ] Training progress visualization
- [ ] TensorBoard/MLflow integration
- [ ] Model export and HuggingFace push

#### 6.2 Embedding Visualization (MagPy)

Interactive 2D/3D scatter plots:

- [ ] UMAP/t-SNE projection display
- [ ] Color by cluster, species, or custom label
- [ ] Click to play associated audio
- [ ] Selection for batch operations

#### 6.3 Services Browser (MagPy)

Unified interface for external APIs:

- [ ] iNaturalist observation search and download
- [ ] Xeno-canto recording search
- [ ] eBird checklist integration
- [ ] Download queue manager

---

## Part 4: Integration Patterns

### 4.1 MagPy → bioamla Direct API

For simple operations, call bioamla functions directly:

```python
# magpy/controllers/audio_controller.py
from bioamla.core.audio import Audio
from bioamla.core.analysis.indices import calculate_indices

class AudioController:
    def calculate_indices(self, indices: list[str]) -> dict:
        return calculate_indices(self._audio.path, indices)
```

### 4.2 Worker Threads for Long Operations

Batch processing and ML inference run in QThread:

```python
# magpy/workers/detection_worker.py
from PySide6.QtCore import QThread, Signal

class DetectionWorker(QThread):
    progress = Signal(int, int)  # current, total
    result = Signal(object)
    error = Signal(str)

    def __init__(self, files: list[str], config: dict):
        super().__init__()
        self.files = files
        self.config = config

    def run(self):
        try:
            from bioamla.core.ml.inference import predict_batch
            for i, result in enumerate(predict_batch(self.files, **self.config)):
                self.progress.emit(i + 1, len(self.files))
            self.result.emit(results)
        except Exception as e:
            self.error.emit(str(e))
```

### 4.3 CLI Bridge for Reproducibility

Generate equivalent CLI commands for operations:

```python
# magpy/controllers/cli_bridge.py
class CLIBridge:
    def generate_command(self, operation: str, **kwargs) -> str:
        """Generate bioamla CLI command for reproducibility."""
        if operation == "detect":
            return f"bioamla ast predict {kwargs['input']} --model-path {kwargs['model']} --output {kwargs['output']}"
        ...
```

---

## Part 5: Technology Stack

### bioamla Core

| Purpose          | Library                          |
| ---------------- | -------------------------------- |
| CLI framework    | Click                            |
| Configuration    | tomllib + tomli-w                |
| Database ORM     | SQLModel (SQLAlchemy + Pydantic) |
| Audio I/O        | soundfile, librosa               |
| Audio playback   | sounddevice                      |
| ML framework     | PyTorch, transformers            |
| Acoustic indices | scikit-maad                      |
| Clustering       | umap-learn, hdbscan              |
| Progress/output  | Rich                             |
| LLM inference    | llama-cpp-python                 |
| RAG              | ChromaDB, sentence-transformers  |

### MagPy GUI

| Purpose             | Library         |
| ------------------- | --------------- |
| GUI framework       | PyQt6 / PySide6 |
| Plotting            | PyQtGraph       |
| Audio playback      | sounddevice     |
| Syntax highlighting | Pygments        |

---

## Part 6: Implementation Timeline

```
2025
│
├── Q1 (Weeks 1-12): Foundation + Analysis Expansion
│   ├── Weeks 1-4: Project system, database layer, command pattern
│   ├── Weeks 5-8: Acoustic indices, embedding extraction
│   └── Weeks 9-12: Clustering, RIBBIT, Xeno-canto
│
├── Q2 (Weeks 13-24): MagPy Core + Annotation
│   ├── Weeks 13-16: Application shell, spectral analyzer
│   ├── Weeks 17-20: Audio controller, terminal screen
│   └── Weeks 21-24: Annotation interface, detection UI
│
├── Q3 (Weeks 25-36): Workflows + Pipeline Editor
│   ├── Weeks 25-28: Detection interface completion
│   ├── Weeks 29-32: Workflow engine (TOML + Jinja2)
│   └── Weeks 33-36: Visual pipeline editor
│
└── Q4 (Weeks 37-52): LLM + Advanced Features
    ├── Weeks 37-40: RAG infrastructure, command generation
    ├── Weeks 41-44: LLM assistant screen
    └── Weeks 45-52: Training wizard, embedding viz, polish
```

---

## Part 7: Testing Strategy

### Unit Tests

- Command execution and undo correctness
- Repository CRUD operations
- Workflow parser validation
- LLM command validation

### Integration Tests

- CLI → service layer → database pipeline
- GUI action → bioamla API → display update
- Workflow execution end-to-end

### System Tests

- Full detection pipeline on sample dataset
- Workflow import/export round-trip
- Memory usage under sustained operation

---

## Part 8: Prototype → Production Workflow

The development cycle for each feature:

```
1. PROTOTYPE (MagPy)
   └── Rapid iteration in GUI context
   └── Manual testing with real data
   └── User feedback collection

2. EXTRACT (bioamla)
   └── Identify stable API surface
   └── Write comprehensive tests
   └── Document public interfaces

3. INTEGRATE (MagPy)
   └── Replace prototype code with bioamla import
   └── Add GUI-specific enhancements
   └── Verify behavior matches prototype

4. RELEASE
   └── bioamla CLI/API available
   └── MagPy GUI wraps bioamla
   └── Documentation updated
```

---

## Appendix A: File Structure After Implementation

```
jmcmeen/bioamla/
├── pyproject.toml
├── src/bioamla/
│   ├── __init__.py
│   ├── core/                     # Core functionality (API layer)
│   │   ├── __init__.py
│   │   ├── config.py             # Extended config system
│   │   ├── project.py            # Project management
│   │   ├── command_log.py        # Command history
│   │   ├── base_api.py           # @config_aware decorator
│   │   ├── audio/                # Audio processing
│   │   │   ├── __init__.py
│   │   │   ├── audio.py
│   │   │   ├── signal.py
│   │   │   ├── augment.py
│   │   │   ├── playback.py
│   │   │   └── torchaudio.py
│   │   ├── ml/                   # Machine learning
│   │   │   ├── __init__.py
│   │   │   ├── inference.py
│   │   │   ├── training.py
│   │   │   ├── embeddings.py     # NEW
│   │   │   ├── ast_model.py
│   │   │   ├── birdnet.py
│   │   │   ├── opensoundscape.py
│   │   │   ├── trainer.py
│   │   │   └── evaluate.py
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── indices.py        # Acoustic indices
│   │   │   ├── clustering.py     # UMAP + HDBSCAN
│   │   │   └── explore.py
│   │   ├── detection/
│   │   │   ├── __init__.py
│   │   │   ├── ast.py
│   │   │   ├── detectors.py
│   │   │   └── ribbit.py         # NEW: OpenSoundscape
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── inaturalist.py
│   │   │   ├── xeno_canto.py
│   │   │   ├── macaulay.py
│   │   │   ├── species.py
│   │   │   └── integrations.py
│   │   ├── files/
│   │   │   ├── __init__.py
│   │   │   ├── io.py
│   │   │   ├── discovery.py
│   │   │   ├── downloads.py
│   │   │   └── paths.py
│   │   ├── workflow/
│   │   │   ├── __init__.py
│   │   │   ├── parser.py         # NEW
│   │   │   ├── engine.py         # NEW
│   │   │   └── validator.py      # NEW
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── rag.py            # NEW
│   │       └── generator.py      # NEW
│   ├── commands/                 # Command pattern (external integration)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── audio.py
│   │   └── analysis.py
│   ├── database/                 # Database layer
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── models.py             # SQLModel entities
│   │   ├── repository.py         # Generic repository
│   │   ├── unit_of_work.py       # UoW pattern
│   │   └── repositories/         # Concrete repositories
│   ├── views/                    # Interface layer (CLI)
│   │   ├── __init__.py
│   │   └── cli.py                # Click CLI entry
│   └── _internal/                # Internal resources
│       ├── templates/
│       │   ├── default.toml
│       │   ├── research.toml
│       │   └── production.toml
│       └── examples/
└── tests/

jmcmeen/magpy/
├── pyproject.toml
├── src/magpy/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── main_window.py
│   ├── screens/
│   │   ├── spectral_analyzer.py
│   │   ├── dataset_browser.py
│   │   ├── terminal.py
│   │   ├── pipeline_editor.py
│   │   ├── command_builder.py
│   │   ├── services_browser.py
│   │   └── llm_assistant.py
│   ├── widgets/
│   │   ├── waveform.py
│   │   ├── spectrogram.py
│   │   ├── annotation_table.py
│   │   └── transport.py
│   ├── controllers/
│   │   ├── audio_controller.py
│   │   ├── project_controller.py
│   │   └── workflow_controller.py
│   ├── workers/
│   │   ├── detection_worker.py
│   │   ├── training_worker.py
│   │   └── indexing_worker.py
│   └── resources/
│       ├── icons/
│       └── styles/
└── tests/
```

---

## Appendix B: Hardware Requirements

| Task                       | Minimum            | Recommended                           |
| -------------------------- | ------------------ | ------------------------------------- |
| bioamla CLI (inference)    | 8GB RAM, CPU       | 16GB RAM, GPU 6GB VRAM                |
| MagPy GUI                  | 8GB RAM            | 16GB RAM, GPU for smooth spectrograms |
| LLM inference (Qwen 7B Q4) | 8GB RAM, ~5GB VRAM | 16GB RAM, RTX 3060+                   |
| LLM fine-tuning            | RTX 3090 24GB      | RTX 4090 24GB                         |

---

*Document Version: 1.0*  
*Created: 2025*  
*For: bioamla + magpy ecosystem*
