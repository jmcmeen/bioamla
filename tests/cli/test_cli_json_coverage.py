"""Every CLI command follows the `--json` contract.

Each command must report a result (``emit_result``, ``emit_file_result`` or a
helper built on them) or be listed in ``NO_JSON_RESULT`` with the reason. A new
command that does neither fails here, so it can't skip the contract unnoticed.
"""

import inspect
import re

import click
import pytest

from bioamla.cli.cli import cli

# Calls that report a --json result (directly or through a shared helper),
# matched as whole names so e.g. build_report( doesn't count as _report(.
# This is a static check: it can't see a branch that returns without
# reporting. Such a branch still gets "result": null, never broken output.
REPORTING_CALLS = re.compile(
    r"(?<![\w.])(emit_result|emit_file_result|fail"
    r"|_report"  # batch commands: the BatchResult
    r"|_run_detect"  # batch detect commands: through _report
    r")\("
)

# Commands with nothing to report, and why. JSON mode is still honoured: they
# get "result": null, or refuse with an error.
NO_JSON_RESULT = {
    "audio play": "plays sound; refuses --json with an error",
}


def _commands(group: click.Group, prefix: str = ""):
    for name, command in sorted(group.commands.items()):
        path = f"{prefix}{name}"
        if isinstance(command, click.Group):
            yield from _commands(command, f"{path} ")
        else:
            yield path, command


ALL_COMMANDS = list(_commands(cli))


def test_the_tree_is_walked() -> None:
    assert len(ALL_COMMANDS) > 90


def _reports(command: click.Command) -> bool:
    source = inspect.getsource(inspect.unwrap(command.callback))
    return bool(REPORTING_CALLS.search(source))


@pytest.mark.parametrize(("path", "command"), ALL_COMMANDS, ids=[p for p, _ in ALL_COMMANDS])
def test_command_reports_a_json_result(path: str, command: click.Command) -> None:
    if path in NO_JSON_RESULT:
        return
    assert _reports(command), (
        f"`bioamla {path}` reports no --json result. Call emit_result() (or a helper "
        "built on it), or add it to NO_JSON_RESULT with the reason."
    )


def test_no_json_list_only_names_real_commands() -> None:
    assert set(NO_JSON_RESULT) <= {path for path, _ in ALL_COMMANDS}


def test_a_command_without_a_result_is_caught() -> None:
    @click.command()
    def silent() -> None:
        click.echo("done")

    @click.command()
    def lookalike() -> None:
        build_report()  # noqa: F821 - only its name matters here

    assert not _reports(silent)
    assert not _reports(lookalike)
