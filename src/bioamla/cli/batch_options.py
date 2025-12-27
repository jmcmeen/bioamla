"""Reusable Click decorators for batch command options."""

from functools import wraps
from typing import Callable

import click


def batch_input_options(f: Callable) -> Callable:
    """Add mutually exclusive batch input options to command.

    Adds:
    - --input-dir: Directory-based batch processing
    - --input-file: CSV-based batch processing

    Validates mutual exclusivity.

    Args:
        f: Click command function

    Returns:
        Decorated function with input options
    """

    @click.option(
        "--input-dir",
        type=click.Path(exists=True, file_okay=False, dir_okay=True),
        help="Input directory for batch processing",
    )
    @click.option(
        "--input-file",
        type=click.Path(exists=True, file_okay=True, dir_okay=False),
        help="Input metadata CSV file (must have file_name column)",
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        input_dir = kwargs.get("input_dir")
        input_file = kwargs.get("input_file")

        if input_dir and input_file:
            raise click.UsageError("--input-dir and --input-file are mutually exclusive")
        if not input_dir and not input_file:
            raise click.UsageError("Either --input-dir or --input-file must be specified")

        return f(*args, **kwargs)

    return wrapper


def batch_output_options(f: Callable) -> Callable:
    """Add batch output options.

    Adds:
    - --output-dir: Output directory (optional for CSV mode, can be in-place)

    Args:
        f: Click command function

    Returns:
        Decorated function with output options
    """

    @click.option("--output-dir", type=click.Path(), help="Output directory")
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper
