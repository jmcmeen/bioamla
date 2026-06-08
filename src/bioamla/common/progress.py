"""
Batch Progress Utilities
========================

Stdlib-only helpers for reporting progress during batch operations and a
free ``process_batch`` function adapted from the old ``BaseService._process_batch``.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, TypeVar

T = TypeVar("T")
I = TypeVar("I")  # noqa: E741 - item type


@dataclass
class BatchProgress:
    """Progress information for batch operations."""

    total: int
    completed: int = 0
    current_file: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        """Get completion percentage."""
        return (self.completed / self.total * 100) if self.total > 0 else 0

    @property
    def remaining(self) -> int:
        """Get remaining items."""
        return self.total - self.completed


# Type alias for progress callback
ProgressCallback = Callable[[BatchProgress], None]


@dataclass
class BatchProcessResult:
    """Result of :func:`process_batch`.

    Attributes:
        results: Successful per-item results, in the order items succeeded.
        errors: List of ``(item, exception)`` tuples for items that failed.
    """

    results: List[T] = field(default_factory=list)
    errors: List[Tuple[object, Exception]] = field(default_factory=list)


def process_batch(
    items: List[I],
    processor: Callable[[I], T],
    on_progress: Optional[ProgressCallback] = None,
    continue_on_error: bool = True,
) -> BatchProcessResult:
    """
    Process a batch of items sequentially with progress reporting.

    Iterates ``items``, calls ``processor(item)`` for each, collects results,
    and reports progress via ``on_progress``. On a per-item error it either
    continues (collecting the ``(item, error)`` pair) or re-raises depending
    on ``continue_on_error``.

    Args:
        items: Items to process.
        processor: Callable applied to each item.
        on_progress: Optional callback invoked with a :class:`BatchProgress`
            before starting, before each item, and after each item completes.
        continue_on_error: If True, collect errors and keep going; if False,
            re-raise the first exception.

    Returns:
        A :class:`BatchProcessResult` with successful results and collected errors.
    """
    progress = BatchProgress(total=len(items))
    if on_progress:
        on_progress(progress)

    result = BatchProcessResult()

    for item in items:
        progress.current_file = str(item)
        if on_progress:
            on_progress(progress)

        try:
            result.results.append(processor(item))
        except Exception as e:
            progress.errors.append(f"{item}: {e}")
            result.errors.append((item, e))
            if not continue_on_error:
                raise

        progress.completed += 1
        if on_progress:
            on_progress(progress)

    return result


__all__ = [
    "BatchProgress",
    "ProgressCallback",
    "BatchProcessResult",
    "process_batch",
]
