"""Central CLI error handling for the bioamla command-line interface."""

import sys

import click

from bioamla.exceptions import BioamlaError


def handle_cli_error(exc: BioamlaError) -> None:
    """Print a friendly error message and exit with status 1."""
    click.echo(f"Error: {exc}", err=True)
    sys.exit(1)
