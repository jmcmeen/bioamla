"""Central CLI error handling for the bioamla command-line interface."""

import sys

from bioamla.exceptions import BioamlaError


def handle_cli_error(exc: BioamlaError) -> None:
    """Print a friendly error message and exit with status 1."""
    from bioamla.cli.console import print_error

    print_error(str(exc))
    sys.exit(1)
