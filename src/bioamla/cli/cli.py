"""
BioAMLA CLI - Bioacoustic Machine Learning Applications
"""

import logging
import sys
import time

import click

from bioamla import __version__
from bioamla.cli.errors import handle_cli_error
from bioamla.cli.logging_setup import LOG_FORMATS, LOG_LEVELS, set_cli_logging
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
    remote,
    serve,
    system,
    util,
    workspace,
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
    from bioamla.cli.output import emit_result

    emit_result({"version": __version__})
    console.print()
    console.print(_gradient_banner(_BANNER, _BANNER_GRADIENT), highlight=False)
    console.print("\n  Bioacoustic & Machine Learning Applications", highlight=False)
    console.print(f"  bioamla [bold]{__version__}[/bold]", highlight=False)
    ctx.exit()


class BioamlaGroup(click.Group):
    """The root group. It runs the ``--json`` contract (:mod:`bioamla.cli.output`).

    Implemented in :meth:`main`, which both the console script and in-process
    callers (``CliRunner``, embedders) go through: JSON mode is switched on and off
    around the whole run, and every way a command can end, including a bare
    ``SystemExit`` or an unexpected exception, yields exactly one envelope.
    """

    #: Marks the root: its options (--json, --log-level, ...) are decided before
    #: settings are read, so settings files and BIOAMLA_* variables can't set them.
    is_bioamla_root = True

    def make_context(self, info_name, args, parent=None, **extra):
        # Option defaults from the user's and the workspace's config.toml and
        # BIOAMLA_<COMMAND>_<OPTION> environment variables (that order, later
        # winning); a flag on the command line still wins over all of them.
        # Done here, inside main()'s error handling, so a malformed settings file
        # is reported like any other error (and as an envelope with --json).
        if parent is None:
            from bioamla.cli.settings_map import settings_default_map

            words = [a for a in args if not a.startswith("-")]
            # Commands that inspect or fix settings, or describe the CLI, still run
            # with a broken settings file (it's reported as a warning).
            lenient = any(a in ("--help", "--version") for a in args) or words[:2] in (
                ["workspace", "config"],
                ["system", "commands"],
            )
            extra.setdefault("default_map", settings_default_map(self, lenient=lenient))
        return super().make_context(info_name, args, parent=parent, **extra)

    def main(self, args=None, prog_name=None, complete_var=None, standalone_mode=True, **extra):
        from bioamla.cli import history, output

        args = list(sys.argv[1:] if args is None else args)
        if not history.should_record(self, args):
            return self._main(args, prog_name, complete_var, standalone_mode, **extra)

        # Record the run in the workspace history, however it ends. Run Click in
        # non-standalone mode so every outcome reaches us as a value or an exception
        # (standalone Click would print and exit before we saw it), then finish the
        # way standalone mode would.
        output.reset_run_record()
        json_run = output.json_flag_given(self, args)
        started = time.time()
        ok, exit_code, error = True, 0, None
        rv = None
        try:
            rv = self._main(args, prog_name, complete_var, False, **extra)
            if isinstance(rv, int) and not isinstance(rv, bool) and rv != 0:  # ctx.exit(n)
                ok, exit_code = False, rv
        except BaseException as exc:
            ok, exit_code, error = False, *_describe_exit(exc)
            if exit_code == 0:
                ok = True
            if not standalone_mode:
                raise
            _finish_standalone_error(exc, json_run)
        finally:
            if error is None and not ok:
                error = output.last_error()  # e.g. a --json run that already reported it
            try:
                history.record_run(
                    self, args, started=started, ended=time.time(), ok=ok,
                    exit_code=exit_code, error=error,
                )  # fmt: skip
            except Exception as e:  # noqa: BLE001 - history must never fail a command
                logging.getLogger("bioamla.cli").warning("Couldn't record this run: %s", e)
            output.reset_run_record()
        if not standalone_mode:
            return rv
        sys.exit(exit_code)

    def _main(self, args, prog_name, complete_var, standalone_mode, **extra):
        from bioamla.cli import output

        if not output.json_flag_given(self, args):
            return super().main(args, prog_name, complete_var, standalone_mode, **extra)

        command = output.command_from_args(self, args)
        output.enable_json_mode()
        try:
            try:
                rv = super().main(args, prog_name, complete_var, standalone_mode=False, **extra)
            except BaseException as exc:
                output.emit_failure(command, exc)
                if standalone_mode and isinstance(exc, (BioamlaError, click.ClickException)):
                    # The JSON is out; show the error for people on stderr, as usual.
                    if isinstance(exc, click.ClickException):
                        exc.show()
                        sys.exit(exc.exit_code)
                    handle_cli_error(exc)
                if standalone_mode and isinstance(exc, (click.exceptions.Abort, KeyboardInterrupt)):
                    sys.exit(1)
                raise
            if isinstance(rv, int) and not isinstance(rv, bool) and rv != 0:  # a ctx.exit() code
                output.emit_failure(command, SystemExit(rv))
            output.finish(command)
            if standalone_mode:
                sys.exit(rv if isinstance(rv, int) and not isinstance(rv, bool) else 0)
            return rv
        finally:
            output.disable_json_mode()


def _describe_exit(exc: BaseException) -> tuple[int, dict | None]:
    """Exit code and recorded error for how a run ended."""
    if isinstance(exc, SystemExit):
        code = exc.code
        if code is None:
            return 0, None
        if isinstance(code, bool) or not isinstance(code, int):
            return 1, None
        return code, None
    if isinstance(exc, click.exceptions.Exit):
        return exc.exit_code, None
    if isinstance(exc, click.ClickException):
        cause = exc.__cause__ if isinstance(exc.__cause__, BioamlaError) else exc
        return exc.exit_code, {"type": type(cause).__name__, "message": exc.format_message()}
    if isinstance(exc, (click.exceptions.Abort, KeyboardInterrupt)):
        return 1, {"type": "Aborted", "message": "Interrupted"}
    return 1, {"type": type(exc).__name__, "message": str(exc)}


def _finish_standalone_error(exc: BaseException, json_run: bool) -> None:
    """End the process the way Click's standalone mode (and the JSON contract) would
    for ``exc``; re-raise what standalone mode lets through."""
    if isinstance(exc, SystemExit):
        raise exc
    if isinstance(exc, click.exceptions.Exit):
        sys.exit(exc.exit_code)
    if isinstance(exc, click.ClickException):
        exc.show()
        sys.exit(exc.exit_code)
    if isinstance(exc, (click.exceptions.Abort, KeyboardInterrupt)):
        if not json_run:
            click.echo("Aborted!", err=True)
        sys.exit(1)
    if json_run and isinstance(exc, BioamlaError):
        handle_cli_error(exc)  # the JSON is out; show the error for people too
    raise exc


@click.group(cls=BioamlaGroup)
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_print_version,
    help="Show the version and exit.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Machine-readable output: one JSON document on stdout, everything else on stderr.",
)
@click.option(
    "--log-format",
    type=click.Choice(LOG_FORMATS),
    default="text",
    show_default=True,
    help="Log output: plain text, or JSON lines on stderr with structured fields.",
)
@click.option(
    "--log-level",
    type=click.Choice(LOG_LEVELS),
    default=None,
    help="Log level for library messages (default: each command's own).",
)
def cli(json_output: bool, log_format: str, log_level: str | None) -> None:
    """BioAMLA - Bioacoustic & Machine Learning Applications

    Use 'bioamla COMMAND --help' for more information on a command.
    """
    # --json is handled by BioamlaGroup.main, before parsing; the option is here so
    # it parses and shows in --help.
    set_cli_logging(log_format, log_level)


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
cli.add_command(remote)
cli.add_command(serve)
cli.add_command(system)
cli.add_command(util)
cli.add_command(workspace)


def main() -> None:
    """CLI entry point with central :class:`BioamlaError` handling.

    Runs the Click group in non-standalone mode so we can catch the whole
    bioamla error family and print a single friendly ``Error: ...`` message
    (exit 1). Click's own usage/abort errors are handled with their normal
    exit codes.
    """
    try:
        rv = cli.main(standalone_mode=False)
    except BioamlaError as e:
        handle_cli_error(e)
    except click.ClickException as e:
        e.show()
        raise SystemExit(e.exit_code) from e
    except (click.exceptions.Abort, KeyboardInterrupt) as e:
        raise SystemExit(1) from e
    except ModuleNotFoundError as e:
        # A library that comes with an extra (bioamla[audio], [ml], ...) isn't
        # installed: say which extra, instead of a traceback.
        from bioamla._extras import extra_hint
        from bioamla.cli.console import print_error

        hint = extra_hint(e)
        if hint is None:
            raise
        print_error(f"{e}. {hint}")
        raise SystemExit(1) from e
    # In non-standalone mode Click returns a ctx.exit() code instead of exiting.
    if isinstance(rv, int) and not isinstance(rv, bool) and rv != 0:
        raise SystemExit(rv)


if __name__ == "__main__":
    main()
