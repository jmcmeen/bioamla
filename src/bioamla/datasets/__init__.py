"""Datasets domain: annotations + dataset operations.

This package folds the old ``services.dataset`` / ``services.annotation`` /
``core.annotations`` layers into plain, exception-raising functions with direct
``pathlib`` I/O.

It covers two related areas:

Annotations
    The :class:`Annotation` data structure, format conversion (Raven selection
    tables, CSV, JSON, Parquet), label engineering (label maps, one-hot/frame
    labels, remapping, filtering), clip extraction, and acoustic measurements.

Datasets
    Merging multiple audio datasets, audio augmentation, license/attribution
    generation, and metadata statistics.

Example:
    >>> from bioamla.datasets import load_csv_annotations, save_raven_selection_table
    >>> anns = load_csv_annotations("annotations.csv")
    >>> save_raven_selection_table(anns, "selections.txt")
"""

from bioamla.datasets.annotation_utils import (
    AnnotationSet,
    annotations_to_one_hot,
    create_label_map,
    filter_labels,
    generate_clip_labels,
    generate_frame_labels,
    load_label_mapping,
    remap_labels,
    save_label_mapping,
)
from bioamla.datasets.annotations import (
    BIOAMLA_ANNOTATION_FORMAT,
    RAVEN_COLUMN_MAP,
    Annotation,
    AnnotationResult,
    create_annotation,
    get_unique_labels,
    load_annotations_from_directory,
    load_bioamla_annotations,
    load_csv_annotations,
    load_raven_selection_table,
    predictions_to_annotations,
    save_bioamla_annotations,
    save_csv_annotations,
    save_json_annotations,
    save_parquet_annotations,
    save_raven_selection_table,
    summarize_annotations,
)
from bioamla.datasets.augmentation import (
    AugmentationConfig,
    augment_audio,
    batch_augment,
    create_augmentation_pipeline,
)
from bioamla.datasets.batch import batch_convert_annotations
from bioamla.datasets.clip_extraction import extract_audio_clips
from bioamla.datasets.labeled_dataset import extract_labeled_dataset
from bioamla.datasets.licenses import (
    generate_license_for_dataset,
    generate_licenses_for_directory,
)
from bioamla.datasets.manifest import (
    BIOAMLA_DATASET_FORMAT,
    DatasetManifest,
    build_dataset_card,
    build_manifest_from_metadata,
    load_dataset_manifest,
    save_dataset_manifest,
    write_dataset_card,
)
from bioamla.datasets.measurements import compute_measurements
from bioamla.datasets.merge import find_species_name, merge_datasets
from bioamla.datasets.partition import partition_dataset
from bioamla.datasets.stats import get_dataset_stats
from bioamla.exceptions import (
    AnnotationError,
    AugmentationError,
    DatasetError,
    LicenseGenerationError,
    MergeError,
)

__all__ = [
    # Annotation data structures
    "Annotation",
    "AnnotationResult",
    "AnnotationSet",
    "RAVEN_COLUMN_MAP",
    # Annotation I/O
    "load_raven_selection_table",
    "save_raven_selection_table",
    "load_csv_annotations",
    "save_csv_annotations",
    "save_json_annotations",
    "save_parquet_annotations",
    "load_bioamla_annotations",
    "save_bioamla_annotations",
    "BIOAMLA_ANNOTATION_FORMAT",
    "load_annotations_from_directory",
    "create_annotation",
    "predictions_to_annotations",
    # Annotation summarization / labels
    "get_unique_labels",
    "summarize_annotations",
    "create_label_map",
    "annotations_to_one_hot",
    "generate_clip_labels",
    "generate_frame_labels",
    "remap_labels",
    "filter_labels",
    "load_label_mapping",
    "save_label_mapping",
    # Clip extraction & measurements
    "extract_audio_clips",
    "extract_labeled_dataset",
    "compute_measurements",
    # Dataset operations
    "merge_datasets",
    "find_species_name",
    "get_dataset_stats",
    "partition_dataset",
    # Dataset manifest
    "DatasetManifest",
    "BIOAMLA_DATASET_FORMAT",
    "build_manifest_from_metadata",
    "save_dataset_manifest",
    "load_dataset_manifest",
    "build_dataset_card",
    "write_dataset_card",
    # Augmentation
    "AugmentationConfig",
    "create_augmentation_pipeline",
    "augment_audio",
    "batch_augment",
    # License generation
    "generate_license_for_dataset",
    "generate_licenses_for_directory",
    # Batch helpers
    "batch_convert_annotations",
    # Exceptions
    "DatasetError",
    "MergeError",
    "AugmentationError",
    "LicenseGenerationError",
    "AnnotationError",
]
