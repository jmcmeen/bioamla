"""
BioAmla - Bioacoustics & Machine Learning Applications
======================================================

A flat, domain-oriented library for bioacoustic analysis. The public surface is
organized into domain subpackages, all importable without pulling in heavy
optional dependencies (torch/transformers, opensoundscape, umap/hdbscan/sklearn,
sounddevice) — those are imported lazily inside the functions that need them and
raise :class:`bioamla.exceptions.DependencyError` if the relevant extra is not
installed.

Domains:
- :mod:`bioamla.audio` — audio I/O, analysis, signal processing, playback.
- :mod:`bioamla.viz` — spectrograms and waveform visualizations.
- :mod:`bioamla.indices` — acoustic indices (ACI, ADI, NDSI, ...).
- :mod:`bioamla.detect` — acoustic event detection (energy, RIBBIT, peaks, ...).
- :mod:`bioamla.cluster` — embedding clustering and novelty detection.
- :mod:`bioamla.catalogs` — external catalogs (xeno-canto, iNaturalist, ...).
- :mod:`bioamla.datasets` — dataset merge / augment / annotations.
- :mod:`bioamla.ml` — AST training, inference, and embeddings.
- :mod:`bioamla.system` — configuration, dependency, and environment info.
- :mod:`bioamla.batch` — generic batch engine and config/result types.
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bioamla")
except PackageNotFoundError:  # package not installed (e.g. source checkout)
    __version__ = "0.0.0"

from bioamla.batch import BatchConfig, BatchResult, SegmentInfo

# Exception hierarchy base — catch the whole family with a single except.
from bioamla.exceptions import BioamlaError

# Domain subpackages are loaded LAZILY (PEP 562) so that `import bioamla` and
# `bioamla --help` stay fast: accessing e.g. ``bioamla.audio`` imports it on first
# use, but a bare ``import bioamla`` does not pull in librosa/numba/scipy/etc.
_LAZY_DOMAINS = frozenset(
    {"audio", "viz", "indices", "detect", "cluster", "catalogs", "datasets", "ml", "system"}
)


def __getattr__(name: str):  # noqa: ANN202 - returns a submodule
    if name in _LAZY_DOMAINS:
        module = importlib.import_module(f"bioamla.{name}")
        globals()[name] = module  # cache so subsequent access skips __getattr__
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # noqa: ANN202
    return sorted([*globals().keys(), *_LAZY_DOMAINS])


__all__ = [
    "__version__",
    # Domains
    "audio",
    "viz",
    "indices",
    "detect",
    "cluster",
    "catalogs",
    "datasets",
    "ml",
    "system",
    # Batch types
    "BatchConfig",
    "BatchResult",
    "SegmentInfo",
    # Errors
    "BioamlaError",
]
