"""The command catalog: `bioamla system commands`."""

import json
import subprocess
import sys

import click
import pytest
from click.testing import CliRunner

from bioamla.cli.catalog import command_catalog
from bioamla.cli.cli import cli
from bioamla.cli.settings_map import resolve_key


def all_paths(group: click.Group, prefix: tuple = ()) -> set[str]:
    paths = set()
    for name, command in group.commands.items():
        path = (*prefix, name)
        paths.add(" ".join(path))
        if isinstance(command, click.Group):
            paths |= all_paths(command, path)
    return paths


@pytest.fixture(scope="module")
def catalog() -> dict:
    return command_catalog(cli)


def test_covers_every_command(catalog) -> None:
    assert {c["path"] for c in catalog["commands"]} == all_paths(cli)


def test_stable_and_json_ready(catalog) -> None:
    assert json.dumps(catalog) == json.dumps(command_catalog(cli))


def test_global_options_are_not_settable(catalog) -> None:
    names = {o["name"] for o in catalog["global_options"]}
    assert {"json", "log_level", "version"} <= names
    assert all("setting" not in o for o in catalog["global_options"])


def test_every_setting_key_resolves_and_env_names_are_unique(catalog) -> None:
    envs = []
    for command in catalog["commands"]:
        for option in command["options"]:
            if "setting" in option:
                resolve_key(cli, option["setting"])  # raises if it names no option
                if "env" in option:
                    envs.append(option["env"])
    assert envs and len(envs) == len(set(envs))


def test_detect_energy_entry(catalog) -> None:
    (energy,) = [c for c in catalog["commands"] if c["path"] == "detect energy"]
    assert energy["kind"] == "command" and energy["arguments"][0]["name"] == "file"
    by_name = {o["name"]: o for o in energy["options"]}
    threshold = by_name["threshold_db"]
    assert threshold["type"] == "float" and threshold["default"] == -20.0
    assert threshold["flags"] == ["--threshold-db", "-t"]
    assert threshold["setting"] == "detect.energy.threshold_db"
    assert threshold["env"] == "BIOAMLA_DETECT_ENERGY_THRESHOLD_DB"
    assert by_name["format"]["choices"] == ["table", "json", "csv"]


def test_prefix(catalog) -> None:
    detect = command_catalog(cli, ["detect"])
    assert {c["path"] for c in detect["commands"]} == {"detect"} | {
        p for p in all_paths(cli) if p.startswith("detect ")
    }
    with pytest.raises(click.UsageError):
        command_catalog(cli, ["nope"])


def test_cli(catalog) -> None:
    result = CliRunner().invoke(cli, ["--json", "system", "commands"])
    assert json.loads(result.stdout)["result"]["commands"] == catalog["commands"]
    human = CliRunner().invoke(cli, ["system", "commands", "detect", "energy"])
    assert human.exit_code == 0 and "BIOAMLA_DETECT" in human.output
    bad = CliRunner().invoke(cli, ["system", "commands", "nope"])
    assert bad.exit_code == 2


def test_builds_without_heavy_imports() -> None:
    code = (
        "import sys; from bioamla.cli.cli import cli; from bioamla.cli.catalog import "
        "command_catalog; command_catalog(cli); "
        "print(sorted(m for m in ('numpy', 'pandas', 'torch', 'librosa') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_help_has_no_control_characters(catalog) -> None:
    assert not [c["path"] for c in catalog["commands"] if "\b" in c["help"]]


def test_unset_flags_default_to_false(catalog) -> None:
    (init,) = [c for c in catalog["commands"] if c["path"] == "models ast init-config"]
    (force,) = [o for o in init["options"] if o["name"] == "force"]
    assert force["is_flag"] and force["default"] is False


def test_short_only_options_have_no_env(catalog) -> None:
    (ws,) = [c for c in catalog["commands"] if c["path"] == "workspace"]
    (dash_c,) = [o for o in ws["options"] if o["flags"] == ["-C"]]
    assert "env" not in dash_c and dash_c["setting"] == "workspace.directory"


def test_hidden_command_named_directly_is_described() -> None:
    @click.group()
    def root() -> None: ...

    @root.command(hidden=True)
    def secret() -> None:
        """Hidden but runnable."""

    assert command_catalog(root)["commands"] == []
    (entry,) = command_catalog(root, ["secret"])["commands"]
    assert entry["path"] == "secret" and entry["help"] == "Hidden but runnable."
