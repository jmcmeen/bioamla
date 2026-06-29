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


# Vertical gradient colour stops for the banner (nature → water → sky).
_BANNER_GRADIENT = [(0, 230, 160), (0, 200, 255), (110, 100, 255)]


def _gradient_banner(banner: str, stops: list[tuple[int, int, int]]) -> "object":
    """Build a Rich ``Text`` for ``banner`` with a top-to-bottom colour gradient.

    Each line is tinted with an RGB colour interpolated across ``stops``. Rich
    strips the styling automatically on a non-TTY, so piped output stays plain.
    """
    from rich.text import Text

    def interp(t: float) -> tuple[int, int, int]:
        if len(stops) == 1:
            return stops[0]
        seg = t * (len(stops) - 1)
        i = min(int(seg), len(stops) - 2)
        f = seg - i
        a, b = stops[i], stops[i + 1]
        return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))  # type: ignore[return-value]

    lines = banner.strip("\n").split("\n")
    last = len(lines) - 1
    text = Text()
    for i, line in enumerate(lines):
        r, g, b = interp(i / last if last else 0)
        text.append(line, style=f"bold #{r:02x}{g:02x}{b:02x}")
        if i < last:
            text.append("\n")
    return text


def _print_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    """Print the BIOAMLA banner and version, then exit."""
    if not value or ctx.resilient_parsing:
        return
    from bioamla.cli.console import console

    console.print()
    console.print(_gradient_banner(_BANNER, _BANNER_GRADIENT), highlight=False)
    console.print(f"\n  Bioacoustic & Machine Learning Applications", highlight=False)
    console.print(f"  bioamla [bold]{__version__}[/bold]", highlight=False)
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
