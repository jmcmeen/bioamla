"""
BioAmla - Bioacoustics & Machine Learning Applications
======================================================

A flat, domain-oriented library for bioacoustic analysis. The public surface is
organized into domain subpackages. Heavy dependencies (torch/transformers,
umap/hdbscan/sklearn, librosa, sounddevice) come with extras and are imported
only inside the functions that need them, so ``import bioamla`` stays fast.

On import, the catalog keys (``EBIRD_API_KEY``, ``XC_API_KEY``, ``HF_TOKEN``) are
read from the nearest ``.env`` file in the working directory or a parent, unless
they're already set. Nothing else in the file is loaded.

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
- :mod:`bioamla.workspace` — project workspaces: artifacts, settings, run history.
- :mod:`bioamla.client` — a client for a bioamla server (``bioamla serve``).
- :mod:`bioamla.operations` — the index of operations the CLI and the server run.
"""

import importlib
import logging
import os
from importlib.metadata import PackageNotFoundError, version

# Exception hierarchy base — catch the whole family with a single except.
from bioamla.exceptions import BioamlaError

try:
    __version__ = version("bioamla")
except PackageNotFoundError:  # package not installed (e.g. source checkout)
    __version__ = "0.0.0"

#: The keys read from a ``.env`` file on import (see :func:`_load_dotenv_keys`).
_DOTENV_KEYS = ("EBIRD_API_KEY", "XC_API_KEY", "HF_TOKEN")


def _load_dotenv_keys() -> None:
    """Set the catalog keys from the nearest ``.env`` in the working directory or a
    parent, unless they're already set. Only these keys: a ``.env`` of another
    project's secrets, or one setting ``BIOAMLA_*`` options, is left alone. A working
    directory that no longer exists means there's no file to read, and a file that
    can't be read (someone else's, not UTF-8, not a regular file) is skipped with a
    warning rather than breaking the import."""
    from dotenv import dotenv_values, find_dotenv

    path = ""
    try:
        path = find_dotenv(usecwd=True)  # from the working directory, not the package
        if not path or not os.path.isfile(path):  # isfile: not a FIFO, which open() blocks on
            return
        values = dotenv_values(path)
    except (OSError, UnicodeDecodeError) as e:
        logging.getLogger(__name__).warning("Couldn't read %s: %s", path or ".env", e)
        return
    for key in _DOTENV_KEYS:
        if values.get(key) and key not in os.environ:
            os.environ[key] = values[key]


_load_dotenv_keys()

# Domain subpackages are loaded LAZILY (PEP 562) so that `import bioamla` and
# `bioamla --help` stay fast: accessing e.g. ``bioamla.audio`` imports it on first
# use, but a bare ``import bioamla`` does not pull in librosa/numba/scipy/etc.
_LAZY_DOMAINS = frozenset(
    {
        "audio",
        "viz",
        "indices",
        "detect",
        "cluster",
        "catalogs",
        "datasets",
        "ml",
        "system",
        "workspace",
    }
)


# The batch types are lazy too: BatchConfig is a validated config, and importing
# pydantic for it would slow every `import bioamla` and CLI start.
_LAZY_BATCH_TYPES = frozenset({"BatchConfig", "BatchResult", "SegmentInfo"})


def __getattr__(name: str):  # noqa: ANN202 - returns a submodule or class
    if name in _LAZY_DOMAINS:
        module = importlib.import_module(f"bioamla.{name}")
        globals()[name] = module  # cache so subsequent access skips __getattr__
        return module
    if name in _LAZY_BATCH_TYPES:
        value = getattr(importlib.import_module("bioamla.batch"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # noqa: ANN202
    return sorted({*globals().keys(), *_LAZY_DOMAINS, *_LAZY_BATCH_TYPES})


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
    "workspace",
    # Batch types
    "BatchConfig",
    "BatchResult",
    "SegmentInfo",
    # Errors
    "BioamlaError",
]
