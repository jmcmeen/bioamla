"""The interfaces bioamla accepts from callers, in one place.

Functions across the domains take callables and objects supplied by the caller:
progress and cancel hooks, and audio models. This module names each of those
interfaces as a :class:`typing.Protocol`, so a downstream package (a GUI, a plugin)
can see what it has to implement and type-check its implementation. Any object of
the right shape satisfies a protocol; nothing needs to inherit from it.

It imports only the standard library, so ``import bioamla.protocols`` works on the
base install, with no extras.

- :class:`OnProgress`: the ``on_progress(completed, total)`` hook on batch functions.
- :class:`OnBatchProgress`: progress hook of :func:`bioamla.common.progress.process_batch`.
- :class:`OnTrainProgress` and :class:`ShouldStop`: the progress and cancel hooks of
  :func:`bioamla.ml.train_ast`.
- :class:`OnDownloadProgress`: the per-file hook of the iNaturalist downloader.
- :class:`AudioModel`: what an audio classification model provides.
  :class:`bioamla.ml.base.BaseAudioModel` subclasses satisfy it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from bioamla.common.progress import BatchProgress
    from bioamla.ml.ast_service import TrainProgress
    from bioamla.ml.base import ModelBackend, PredictionResult


class OnProgress(Protocol):
    """Called after each item of a batch: ``completed`` of ``total`` items are done.

    Raising from the hook stops the batch, which is how callers cancel one.
    """

    def __call__(self, completed: int, total: int, /) -> None: ...


class OnBatchProgress(Protocol):
    """Called with a :class:`~bioamla.common.progress.BatchProgress` after each item."""

    def __call__(self, progress: BatchProgress, /) -> None: ...


class OnTrainProgress(Protocol):
    """Called with a :class:`~bioamla.ml.TrainProgress` as training advances."""

    def __call__(self, progress: TrainProgress, /) -> None: ...


class ShouldStop(Protocol):
    """Polled during a long run; returning True cancels it."""

    def __call__(self) -> bool: ...


class OnDownloadProgress(Protocol):
    """Called after each download: ``completed`` of ``total``, with the item's name."""

    def __call__(self, completed: int, total: int, name: str, /) -> None: ...


@runtime_checkable
class AudioModel(Protocol):
    """An audio classification model: load weights, then predict on audio or embed it.

    :class:`bioamla.ml.base.BaseAudioModel` subclasses (``ASTModel``, ...) satisfy
    this. ``isinstance(obj, AudioModel)`` checks that the members exist, not their
    signatures.
    """

    @property
    def backend(self) -> ModelBackend:
        """Which model family this is."""
        ...

    @property
    def classes(self) -> list[str]:
        """Class labels, in output order."""
        ...

    @property
    def num_classes(self) -> int:
        """Number of class labels."""
        ...

    def load(self, model_path: str, **kwargs: Any) -> AudioModel:
        """Load weights from a path or hub id and return the model."""
        ...

    def predict(self, audio: Any, sample_rate: int | None = None) -> list[PredictionResult]:
        """Predict on a file path, an array or a tensor (``sample_rate`` for raw audio)."""
        ...

    def predict_file(self, filepath: str) -> list[PredictionResult]:
        """Predict on an audio file."""
        ...

    def extract_embeddings(
        self, audio: Any, sample_rate: int | None = None, layer: str | None = None
    ) -> Any:
        """Return embeddings (a numpy array) from ``layer``, or the model's default."""
        ...


__all__ = [
    "AudioModel",
    "OnBatchProgress",
    "OnDownloadProgress",
    "OnProgress",
    "OnTrainProgress",
    "ShouldStop",
]
