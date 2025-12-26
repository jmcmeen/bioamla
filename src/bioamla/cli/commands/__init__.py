"""CLI command modules for bioamla."""

from .annotation import annotation
from .audio import audio
from .batch import batch
from .catalogs import catalogs
from .cluster import cluster
from .config import config
from .dataset import dataset
from .detect import detect
from .indices import indices
from .models import models

__all__ = [
    "annotation",
    "audio",
    "batch",
    "catalogs",
    "cluster",
    "config",
    "dataset",
    "detect",
    "indices",
    "models",
]
