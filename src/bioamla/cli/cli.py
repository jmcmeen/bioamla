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

_BANNER = r"""
██████╗ ██╗ ██████╗  █████╗ ███╗   ███╗██╗      █████╗
██╔══██╗██║██╔═══██╗██╔══██╗████╗ ████║██║     ██╔══██╗
██████╔╝██║██║   ██║███████║██╔████╔██║██║     ███████║
██╔══██╗██║██║   ██║██╔══██║██║╚██╔╝██║██║     ██╔══██║
██████╔╝██║╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗██║  ██║
╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
"""


def _print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Print the BIOAMLA banner and version, then exit."""
    if not value or ctx.resilient_parsing:
        return
    click.echo(_BANNER)
    click.echo(f"  bioamla {__version__}")
    ctx.exit()


@click.group()
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_version,
    help="Show the version and exit.",
)
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
