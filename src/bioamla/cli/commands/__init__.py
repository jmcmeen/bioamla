"""CLI command modules for bioamla."""

from .annotation import annotation
from .audio import audio
from .batch import batch
from .catalogs import catalogs
from .cluster import cluster
from .dataset import dataset
from .detect import detect
from .indices import indices
from .models import models
from .system import system
from .util import util

__all__ = [
    "annotation",
    "audio",
    "batch",
    "catalogs",
    "cluster",
    "dataset",
    "detect",
    "indices",
    "models",
    "system",
    "util",
]
