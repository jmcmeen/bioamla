"""CLI command modules for bioamla."""

from .annotation import annotation
from .audio import audio
from .batch import batch
from .cluster import cluster
from .config import config
from .dataset import dataset
from .detect import detect
from .examples import examples
from .indices import indices
from .learn import learn
from .models import models
from .pipeline import pipeline
from .realtime import realtime
from .integrations import services

__all__ = [
    "annotation",
    "audio",
    "batch",
    "cluster",
    "config",
    "dataset",
    "detect",
    "examples",
    "indices",
    "learn",
    "models",
    "pipeline",
    "realtime",
    "integrations",
]
