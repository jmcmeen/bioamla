"""
Bioamla example workflows.

This module provides access to example shell scripts that demonstrate
bioamla capabilities. Examples can be accessed via the CLI:

    bioamla examples list              # List all examples
    bioamla examples show 01           # Show example content
    bioamla examples copy 01 ./        # Copy example to directory
    bioamla examples copy-all ./       # Copy all examples
"""

from importlib import resources
from pathlib import Path

# Example metadata: (filename, title, description)
EXAMPLES = {
    "00": (
        "00_starting_a_project.sh",
        "Starting a Project",
        "Initialize a bioamla project and set up directory structure",
    ),
    "01": (
        "01_audio_preprocessing.sh",
        "Audio Preprocessing",
        "Prepare raw recordings: filtering, denoising, normalization, segmentation",
    ),
    "02": (
        "02_acoustic_indices.sh",
        "Acoustic Indices",
        "Compute soundscape ecology metrics: ACI, ADI, AEI, BIO, NDSI",
    ),
    "03": (
        "03_species_detection.sh",
        "Species Detection",
        "Detect vocalizations: energy, RIBBIT, CWT peaks, accelerating patterns",
    ),
    "04": (
        "04_clustering_discovery.sh",
        "Clustering & Discovery",
        "Unsupervised sound discovery: embeddings, UMAP, HDBSCAN, novelty detection",
    ),
    "05": (
        "05_active_learning.sh",
        "Active Learning",
        "Efficient annotation: uncertainty sampling, query batches, simulation",
    ),
    "06": (
        "06_data_acquisition.sh",
        "Data Acquisition",
        "Download from online sources: iNaturalist, Xeno-canto, Macaulay Library",
    ),
    "07": (
        "07_model_training.sh",
        "Model Training",
        "Train ML models: AST fine-tuning with HuggingFace datasets (ESC-50, frogs)",
    ),
    "08": (
        "08_batch_inference.sh",
        "Batch Inference",
        "Large-scale classification with HuggingFace models (AudioSet, ESC-50, frogs)",
    ),
    "09": (
        "09_visualization.sh",
        "Visualization",
        "Generate spectrograms: Mel, STFT, MFCC, waveform, colormaps",
    ),
    "10": (
        "10_end_to_end_survey.sh",
        "End-to-End Survey",
        "Complete wildlife survey workflow with HuggingFace models",
    ),
    # Dataset-specific examples
    "11": (
        "11_esc50_ast_train.sh",
        "ESC-50 Training",
        "Train AST on ESC-50 environmental sounds (ashraq/esc50)",
    ),
    "12": (
        "12_esc50_ast_inference.sh",
        "ESC-50 Inference",
        "Run inference with bioamla/ast-esc50 model",
    ),
    "13": (
        "13_scp_frogs_ast_train.sh",
        "Frogs Training",
        "Train AST on frog species (bioamla/scp-frogs-inat-v1)",
    ),
    "14": (
        "14_scp_frogs_ast_inference.sh",
        "Frogs Inference",
        "Run inference with bioamla/scp-frogs model",
    ),
    "15": (
        "15_birdset_ast_train.sh",
        "BirdSet Training",
        "Train AST on BirdSet bird sounds (samuelstevens/BirdSet)",
    ),
    "16": (
        "16_birdset_ast_inference.sh",
        "BirdSet Inference",
        "Run inference for bird species classification",
    ),
    "17": (
        "17_inat_workflow.sh",
        "iNaturalist Workflow",
        "Download from iNaturalist, train model, run inference",
    ),
    # Testing utility
    "99": (
        "99_run_all.sh",
        "Run All Examples",
        "Execute all example scripts in numeric order for testing",
    ),
}


def get_example_path(example_id: str) -> Path:
    """Get the path to an example file.

    Args:
        example_id: The example ID (e.g., "01", "02") or full filename.

    Returns:
        Path to the example file.

    Raises:
        ValueError: If example_id is not found.
    """
    # Handle both "01" and "01_audio_preprocessing.sh" formats
    if example_id in EXAMPLES:
        filename = EXAMPLES[example_id][0]
    else:
        # Try to find by filename
        filename = example_id
        found = False
        for _, (fname, _, _) in EXAMPLES.items():
            if fname == example_id or fname.startswith(example_id):
                filename = fname
                found = True
                break
        if not found and not any(fname == filename for _, (fname, _, _) in EXAMPLES.items()):
            raise ValueError(f"Example not found: {example_id}")

    # Use importlib.resources to get the file path
    try:
        # Python 3.9+
        ref = resources.files(__package__) / filename
        return Path(str(ref))
    except AttributeError:
        # Python 3.8 fallback
        with resources.path(__package__, filename) as p:
            return p


def get_example_content(example_id: str) -> str:
    """Get the content of an example file.

    Args:
        example_id: The example ID (e.g., "01", "02").

    Returns:
        Content of the example file as a string.
    """
    if example_id in EXAMPLES:
        filename = EXAMPLES[example_id][0]
    else:
        raise ValueError(f"Example not found: {example_id}")

    try:
        # Python 3.9+
        ref = resources.files(__package__) / filename
        return ref.read_text()
    except AttributeError:
        # Python 3.8 fallback
        return resources.read_text(__package__, filename)


def list_examples() -> list[tuple[str, str, str]]:
    """List all available examples.

    Returns:
        List of tuples: (id, title, description)
    """
    return [(eid, title, desc) for eid, (_, title, desc) in sorted(EXAMPLES.items())]


def get_all_example_files() -> list[tuple[str, str]]:
    """Get all example filenames with their IDs.

    Returns:
        List of tuples: (id, filename)
    """
    return [(eid, fname) for eid, (fname, _, _) in sorted(EXAMPLES.items())]
