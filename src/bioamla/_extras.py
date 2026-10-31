"""Optional dependency groups ("extras") and the errors raised when one is missing.

``pip install bioamla`` is the slim base: audio I/O and recorder metadata, file
discovery, the batch engine, dataset tooling and the catalog clients. Heavier
features need an extra (see ``[project.optional-dependencies]``):

- ``audio``: librosa, sounddevice (signal processing, playback)
- ``viz``: matplotlib on top of ``audio`` (spectrograms, plots, videos)
- ``indices``, ``detect``: ``audio`` plus numba for indices
- ``ml``: torch, transformers, datasets, ... (AST models, augmentation, HF datasets)
- ``cluster``: umap, hdbscan, scikit-learn
- ``all``: every extra

The domain packages that need an extra wholesale (``viz``, ``indices``,
``detect``, ``ml``, ``cluster``) call :func:`require_extra` when first
imported. Functions elsewhere import their heavy dependency through
:func:`import_extra`. Either way a missing extra raises
:class:`~bioamla.exceptions.DependencyError` naming the install command.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

from bioamla.exceptions import DependencyError

# Top-level modules each extra provides (import names, not distribution names).
EXTRAS: dict[str, tuple[str, ...]] = {
    "audio": ("librosa", "sounddevice"),
    "viz": ("librosa", "matplotlib"),
    "indices": ("librosa", "numba"),
    "detect": ("librosa",),
    "ml": (
        "torch",
        "torchaudio",
        "transformers",
        "datasets",
        "audiomentations",
        "evaluate",
        "mlflow",
    ),
    "cluster": ("umap", "hdbscan", "sklearn"),
}

# The extra to suggest for a missing module: the smallest one that provides it.
MODULE_EXTRA: dict[str, str] = {}
for _extra, _modules in EXTRAS.items():
    for _module in _modules:
        MODULE_EXTRA.setdefault(_module, _extra)


def install_hint(extra: str) -> str:
    """The command that installs ``extra``."""
    return f"pip install 'bioamla[{extra}]'"


def require_extra(extra: str, feature: str) -> None:
    """Raise unless every module ``extra`` provides can be found (none are imported).

    Args:
        extra: Name of the extra (a key of :data:`EXTRAS`).
        feature: What needs it, for the message (e.g. ``"bioamla.viz"``).

    Raises:
        DependencyError: Naming the missing modules and the install command.
    """
    missing = [m for m in EXTRAS[extra] if importlib.util.find_spec(m) is None]
    if missing:
        raise DependencyError(
            f"{feature} needs the [{extra}] extra (missing: {', '.join(missing)}). "
            f"Install it with: {install_hint(extra)}"
        )


def import_extra(module: str, extra: str | None = None) -> ModuleType:
    """Import ``module``, raising :class:`DependencyError` if its extra isn't installed.

    Args:
        module: Module to import (``"librosa"``, ``"torch.nn"``, ...).
        extra: Extra that provides it; looked up from the module name if omitted.

    Raises:
        DependencyError: If the module (or the package it belongs to) is missing.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        hint = extra_hint(e, extra)
        if hint is None:
            raise
        raise DependencyError(f"{module} is not installed. {hint}") from e


def extra_hint(error: ModuleNotFoundError, extra: str | None = None) -> str | None:
    """Install advice for a missing module that belongs to an extra, else ``None``.

    With ``extra`` given, the module was being imported for that extra, so even a
    missing transitive dependency (librosa's numba, say) is fixed by installing it.
    Otherwise the extra is looked up from the missing module's name.
    """
    if extra is None:
        extra = MODULE_EXTRA.get((error.name or "").split(".")[0])
    if extra is None:
        return None
    return f"It comes with the [{extra}] extra: {install_hint(extra)}"


__all__ = [
    "EXTRAS",
    "MODULE_EXTRA",
    "extra_hint",
    "import_extra",
    "install_hint",
    "require_extra",
]
