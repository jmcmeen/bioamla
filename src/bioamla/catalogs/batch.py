"""Batch helpers for catalog operations.

Thin wrappers over :func:`bioamla.batch.run_batch` so the CLI cut-over can wire
catalog batch commands to functions that already live in this package. Each
helper maps an item (species name) to a per-item catalog call and aggregates the
results into a :class:`~bioamla.batch.BatchResult`.
"""

from collections.abc import Callable

from bioamla.batch import BatchResult, run_batch
from bioamla.catalogs import macaulay, xeno_canto
from bioamla.catalogs import species as species_mod


def batch_download_xeno_canto(
    species_list: list[str],
    output_dir: str = "./xc_recordings",
    quality: str = "A",
    max_recordings: int = 10,
    *,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Download Xeno-canto recordings for each species in ``species_list``."""

    def _process(name: str) -> str:
        xeno_canto.download(
            species=name,
            quality=quality,
            max_recordings=max_recordings,
            output_dir=output_dir,
        )
        return name

    return run_batch(
        species_list,
        _process,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )


def batch_download_macaulay(
    species_codes: list[str],
    output_dir: str = "./ml_recordings",
    min_rating: int = 3,
    max_recordings: int = 10,
    *,
    max_workers: int = 1,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Download Macaulay Library recordings for each eBird species code."""

    def _process(code: str) -> str:
        macaulay.download(
            species_code=code,
            min_rating=min_rating,
            max_recordings=max_recordings,
            output_dir=output_dir,
        )
        return code

    return run_batch(
        species_codes,
        _process,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )


def batch_lookup_species(
    names: list[str],
    ebird_only: bool = False,
    *,
    continue_on_error: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BatchResult:
    """Look up each name via :func:`bioamla.catalogs.species.lookup`.

    Runs sequentially (the taxonomy cache is shared module state).
    """

    def _process(name: str) -> str:
        info = species_mod.lookup(name, ebird_only=ebird_only)
        return info.scientific_name

    return run_batch(
        names,
        _process,
        max_workers=1,
        continue_on_error=continue_on_error,
        on_progress=on_progress,
    )
