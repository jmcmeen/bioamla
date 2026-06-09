"""
The library follows idiomatic Python error handling: functions return plain data
and raise exceptions on failure. Every exception derives from :class:`BioamlaError`,
so applications (and the CLI) can catch the whole family with a single ``except``.

The CLI layer catches :class:`BioamlaError` centrally and prints a friendly message
(see ``bioamla.cli.errors``); library consumers can catch the base class or any of
the more specific subclasses below.
"""

from __future__ import annotations


class BioamlaError(Exception):
    """Base class for all bioamla errors."""


class InvalidInputError(BioamlaError):
    """Caller passed bad, contradictory, or out-of-range arguments."""


class NotFoundError(BioamlaError):
    """A required input path or resource does not exist."""


class AudioLoadError(BioamlaError):
    """Failed to load or decode an audio file."""


class AudioSaveError(BioamlaError):
    """Failed to encode or write an audio file."""


class DependencyError(BioamlaError):
    """A required dependency is unavailable at runtime.

    All Python dependencies ship in the base install, so this is reserved for
    genuine environment problems (e.g. a missing system library or a broken
    install).
    """


class ModelError(BioamlaError):
    """Model load, inference, or embedding-extraction failure."""


class CatalogError(BioamlaError):
    """External catalog / API failure (xeno-canto, iNaturalist, eBird, Macaulay, HF)."""


class SpeciesError(CatalogError):
    """Species lookup, name-conversion, or taxonomy-export failure."""


class ConfigError(BioamlaError):
    """Configuration file parse or validation failure."""


class ProcessingError(BioamlaError):
    """A pipeline/processing failure that doesn't fit a more specific class."""


class ClusteringError(BioamlaError):
    """A clustering, dimensionality-reduction, or novelty-detection failure."""


class DetectionError(BioamlaError):
    """An acoustic detection (energy, RIBBIT, CWT peaks, patterns) failure."""


class InvalidDetectionParams(InvalidInputError):
    """Caller passed invalid parameters to a detector (e.g. high_freq <= low_freq)."""


class DatasetError(BioamlaError):
    """Base class for dataset-domain failures (merge, augment, license, stats)."""


class MergeError(DatasetError):
    """Merging audio datasets failed."""


class AugmentationError(DatasetError):
    """Audio augmentation (noise/stretch/pitch/gain) failed."""


class LicenseGenerationError(DatasetError):
    """Generating a license/attribution file from dataset metadata failed."""


class AnnotationError(BioamlaError):
    """An annotation operation (import/export/extraction/measurement) failed."""


__all__ = [
    "BioamlaError",
    "InvalidInputError",
    "NotFoundError",
    "AudioLoadError",
    "AudioSaveError",
    "DependencyError",
    "ModelError",
    "CatalogError",
    "ConfigError",
    "ProcessingError",
    "SpeciesError",
    "ClusteringError",
    "DetectionError",
    "InvalidDetectionParams",
    "DatasetError",
    "MergeError",
    "AugmentationError",
    "LicenseGenerationError",
    "AnnotationError",
]
