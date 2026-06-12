"""
BioAMLA CLI - Bioacoustic Machine Learning Applications
"""

import click

from bioamla import __version__
from bioamla.cli.errors import handle_cli_error
from bioamla.exceptions import BioamlaError

from .commands import (
    annotation,
    audio,
    batch,
    catalogs,
    cluster,
    dataset,
    detect,
    indices,
    models,
    system,
    util,
)


@click.group()
@click.version_option(version=__version__, prog_name="bioamla")
def cli() -> None:
    """BioAMLA - Bioacoustic & Machine Learning Applications

    Use 'bioamla COMMAND --help' for more information on a command.
    """
    pass


# Register command groups (alphabetical — Click lists them in registration order)
cli.add_command(annotation)
cli.add_command(audio)
cli.add_command(batch)
cli.add_command(catalogs)
cli.add_command(cluster)
cli.add_command(dataset)
cli.add_command(detect)
cli.add_command(indices)
cli.add_command(models)
cli.add_command(system)
cli.add_command(util)


def main() -> None:
    """CLI entry point with central :class:`BioamlaError` handling.

    Runs the Click group in non-standalone mode so we can catch the whole
    bioamla error family and print a single friendly ``Error: ...`` message
    (exit 1). Click's own usage/abort errors are handled with their normal
    exit codes.
    """
    try:
        cli.main(standalone_mode=False)
    except BioamlaError as e:
        handle_cli_error(e)
    except click.ClickException as e:
        e.show()
        raise SystemExit(e.exit_code) from e
    except (click.exceptions.Abort, KeyboardInterrupt) as e:
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
