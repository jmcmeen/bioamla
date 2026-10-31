"""Reusable Click option decorators shared across command groups.

These reproduce options that were previously re-declared inline in every batch
command (``--max-workers``/``--recursive``/``--quiet``), so names, short flags,
defaults, and help stay identical everywhere. Complements the input/output
decorators in :mod:`bioamla.cli.batch_options`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from bioamla.audio.features import FrameFeatureConfig


def max_workers_option(f: Callable) -> Callable:
    """Add the standard ``--max-workers`` / ``-w`` parallelism option."""
    return click.option(
        "--max-workers", "-w", default=1, type=int, help="Number of parallel workers"
    )(f)


def recursive_option(f: Callable) -> Callable:
    """Add the standard ``--recursive/--no-recursive`` directory-search option."""
    return click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")(f)


def quiet_option(f: Callable) -> Callable:
    """Add the standard ``--quiet`` / ``-q`` progress-suppression flag."""
    return click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")(f)


def frame_feature_options(f: Callable) -> Callable:
    """Analysis options shared by ``audio features`` and ``batch audio features``."""
    options = [
        click.option(
            "--trajectory", is_flag=True, help="Add pc1..pc3: the MFCC-PCA timbre trajectory"
        ),
        click.option(
            "--gate-db",
            default=38.0,
            show_default=True,
            type=float,
            help="Trajectory: leave out frames this many dB below the loudest",
        ),
        click.option(
            "--no-denoise", is_flag=True, help="Don't subtract the stationary noise floor"
        ),
        click.option(
            "--n-fft", type=int, default=None, help="FFT size (default 1024; 2048 above 60 kHz)"
        ),
        click.option("--hop", type=int, default=None, help="Hop in samples (default n_fft / 4)"),
        click.option("--n-mels", type=int, default=40, show_default=True, help="Mel bands"),
        click.option(
            "--n-mfcc", type=int, default=13, show_default=True, help="MFCCs, including c0"
        ),
        click.option(
            "--f-min", type=float, default=250.0, show_default=True, help="Band lower edge in Hz"
        ),
        click.option(
            "--f-max",
            type=float,
            default=None,
            help="Band upper edge in Hz (default min(0.92 x Nyquist, 16 kHz))",
        ),
    ]
    for option in reversed(options):
        f = option(f)
    return f


def frame_feature_config(
    no_denoise: bool,
    n_fft: int | None,
    hop: int | None,
    n_mels: int,
    n_mfcc: int,
    f_min: float,
    f_max: float | None,
) -> "FrameFeatureConfig":
    """A FrameFeatureConfig from the shared options (raises InvalidInputError)."""
    from bioamla.audio import FrameFeatureConfig

    return FrameFeatureConfig(
        n_fft=n_fft,
        hop=hop,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        f_min=f_min,
        f_max=f_max,
        denoise=not no_denoise,
    )
