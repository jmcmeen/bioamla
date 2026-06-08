"""Module-level picklable processors for batch tests.

``ProcessPoolExecutor`` (used by :func:`bioamla.batch.run_batch` in parallel
mode) requires top-level functions; closures defined inside test bodies are not
picklable.
"""


def square(x: int) -> int:
    """Return x squared (a picklable success processor)."""
    return x * x


def fail_on_odd(x: int) -> int:
    """Raise for odd inputs, return x for even ones."""
    if x % 2 == 1:
        raise ValueError(f"odd value: {x}")
    return x
