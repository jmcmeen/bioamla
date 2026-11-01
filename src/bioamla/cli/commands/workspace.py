"""Project workspaces: create one, see what's in it, link or import files."""

from pathlib import Path

import click

from bioamla.exceptions import BioamlaError


@click.group()
@click.option(
    "-C",
    "directory",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Run as if started in this directory (like git -C)",
)
@click.pass_context
def workspace(ctx: click.Context, directory: str | None) -> None:
    """Project workspaces: a folder with a .bioamla/ inside for its recordings,
    settings, annotations and run outputs.

    Commands find the workspace by walking up from the current directory.
    """
    ctx.obj = {"start": directory}


def _current(ctx: click.Context):
    """The workspace for this command, or a ClickException saying how to make one."""
    from bioamla.workspace import find_workspace

    start = (ctx.obj or {}).get("start")
    try:
        ws = find_workspace(start)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    if ws is None:
        where = start or "the current directory"
        raise click.ClickException(
            f"Not inside a bioamla workspace ({where}). Create one with `bioamla workspace init`."
        )
    from bioamla.cli.output import note_workspace

    note_workspace(ws)  # this run's history goes to the workspace it worked on
    return ws


def _artifact_row(ws, artifact) -> dict:
    path = ws.resolve(artifact)
    return {**artifact.to_dict(), "resolved": str(path), "exists": path.exists()}


@workspace.command("init")
@click.argument("directory", required=False, default=".")
@click.option("--name", default=None, help="Workspace name (default: the folder's name)")
@click.pass_context
def workspace_init(ctx: click.Context, directory: str, name: str | None) -> None:
    """Make DIRECTORY (default: the current one) a workspace."""
    from bioamla.cli.console import print_kv, print_success
    from bioamla.cli.output import emit_result
    from bioamla.workspace import create_workspace

    start = (ctx.obj or {}).get("start")
    root = Path(start or ".") / directory
    try:
        ws = create_workspace(root, name=name)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    from bioamla.cli.output import note_workspace

    note_workspace(ws)
    if emit_result(ws.to_dict()):
        return
    print_success(f"Created workspace {ws.name!r}")
    print_kv("Root", ws.root)


_STATUS_RUNS = 5


@workspace.command("status")
@click.pass_context
def workspace_status(ctx: click.Context) -> None:
    """Show the workspace: name, location, what it holds and its last few runs."""
    from bioamla.cli.console import print_kv, print_table
    from bioamla.cli.output import emit_result

    ws = _current(ctx)
    try:
        info = ws.to_dict()
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    runs = ws.history(limit=_STATUS_RUNS)
    if emit_result({**info, "last_runs": runs}):
        return
    print_kv("Workspace", info["name"])
    print_kv("Root", info["root"])
    print_kv("Created", info["created"])
    counts = info["artifacts"]
    print_kv(
        "Artifacts",
        ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())) if counts else "none",
    )
    if not runs:
        print_kv("Last runs", "none")
        return
    print_table("Last runs", _RUN_COLUMNS, [_run_row(run) for run in runs])


_KIND_CHOICE = click.Choice(["audio_file", "folder", "dataset", "model", "file"])


def _add_command(command_name: str, copy: bool, verb: str, doc: str):
    """``workspace add`` and ``workspace import``: the same options, linking or copying."""

    @workspace.command(command_name, help=doc)
    @click.argument("paths", nargs=-1, required=True)
    @click.option(
        "--kind", type=_KIND_CHOICE, default=None, help="Artifact kind (default: guessed)"
    )
    @click.option("--name", default=None, help="Display name (only with a single PATH)")
    @click.pass_context
    def command(ctx: click.Context, paths: tuple[str, ...], kind, name) -> None:
        from bioamla.cli.console import print_success
        from bioamla.cli.output import emit_result

        if name is not None and len(paths) > 1:
            raise click.UsageError("--name applies to a single PATH")
        ws = _current(ctx)
        # Relative paths are relative to -C, like git -C.
        base = Path((ctx.obj or {}).get("start") or ".")
        resolved = [p if Path(p).is_absolute() else str(base / p) for p in paths]
        try:
            many = ws.import_many if copy else ws.link_many
            added = [_artifact_row(ws, a) for a in many(resolved, kind=kind, name=name or "")]
        except BioamlaError as e:
            raise click.ClickException(str(e)) from e
        if emit_result({"workspace": str(ws.root), "artifacts": added}):
            return
        for row in added:
            print_success(f"{verb} {row['kind']} {row['path']}  (id {row['id']})")

    return command


workspace_add = _add_command(
    "add",
    copy=False,
    verb="Linked",
    doc="""Link files or folders into the workspace, leaving them where they are.

    All PATHS are checked before any is added.

    \b
    Examples:
        bioamla workspace add recordings/site-a
        bioamla workspace add /data/archive/2024 --kind folder
    """,
)
workspace_import = _add_command(
    "import",
    copy=True,
    verb="Imported",
    doc="""Copy files or folders into the workspace (.bioamla/imported/).

    All PATHS are checked before any is copied; if a copy fails, none is kept.
    """,
)


@workspace.command("list")
@click.option("--kind", type=_KIND_CHOICE, default=None, help="Only this kind")
@click.pass_context
def workspace_list(ctx: click.Context, kind) -> None:
    """List the workspace's artifacts."""
    from bioamla.cli.console import print_table
    from bioamla.cli.output import emit_result

    ws = _current(ctx)
    try:
        rows = [_artifact_row(ws, a) for a in ws.artifacts(kind=kind)]
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    if emit_result({"workspace": str(ws.root), "artifacts": rows}):
        return
    if not rows:
        click.echo("No artifacts. Add some with `bioamla workspace add PATH`.")
        return
    print_table(
        f"Artifacts in {ws.name}",
        ["id", "kind", "mode", "path", "found"],
        [[r["id"], r["kind"], r["mode"], r["path"], "yes" if r["exists"] else "MISSING"]
         for r in rows],
    )  # fmt: skip


@workspace.command("remove")
@click.argument("artifact_id")
@click.option("--delete-import", is_flag=True, help="Also delete the copy of an imported artifact")
@click.pass_context
def workspace_remove(ctx: click.Context, artifact_id: str, delete_import: bool) -> None:
    """Forget an artifact. Linked files are never touched."""
    from bioamla.cli.console import print_success
    from bioamla.cli.output import emit_result

    ws = _current(ctx)
    try:
        artifact = ws.remove(artifact_id, delete_import=delete_import)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    deleted = delete_import and artifact.mode == "imported"
    if emit_result(
        {"workspace": str(ws.root), "removed": artifact.to_dict(), "deleted_copy": deleted}
    ):
        return
    print_success(f"Removed {artifact.kind} {artifact.path}")
    if delete_import and not deleted:
        click.echo("  It was linked, not imported: the file was left where it is.")


@workspace.command("history")
@click.option("--limit", "-n", type=click.IntRange(min=1), default=20, show_default=True,
              help="Show the last N runs")  # fmt: skip
@click.option("--failed", is_flag=True, help="Only runs that failed")
@click.option("--all", "show_all", is_flag=True, help="Every run (ignores --limit)")
@click.pass_context
def workspace_history(ctx: click.Context, limit: int, failed: bool, show_all: bool) -> None:
    """Commands run in this workspace: when, what, and whether they worked.

    Every bioamla command run inside a workspace is recorded in
    .bioamla/history.jsonl (values of secret-looking options are hidden).
    """
    from bioamla.cli.console import print_table
    from bioamla.cli.output import emit_result

    ws = _current(ctx)
    runs = ws.history(limit=None if show_all else limit, failed_only=failed)
    if emit_result({"workspace": str(ws.root), "runs": runs}):
        return
    if not runs:
        click.echo("No runs recorded yet." if not failed else "No failed runs.")
        return
    print_table(f"History of {ws.name}", _RUN_COLUMNS, [_run_row(run) for run in runs])


_RUN_COLUMNS = ["time (UTC)", "command", "result", "took"]


def _run_row(run: dict) -> list[str]:
    """A history record as a table row (hand-edited records may lack fields or hold
    odd types)."""
    status = "ok" if run.get("ok") else f"exit {run.get('exit_code')}"
    argv = run.get("argv")
    command = " ".join(map(str, argv)) if isinstance(argv, list) else str(run.get("command"))
    duration = run.get("duration")
    took = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "?"
    return [str(run.get("time", ""))[:19].replace("T", " "), command, status, took]


@workspace.group("config")
def workspace_config() -> None:
    """Settings: option defaults for commands, from config.toml files.

    A key names a command and one of its options, e.g. `detect.energy.threshold_db`
    for `bioamla detect energy --threshold-db`. Workspace settings (.bioamla/config.toml)
    override user settings (~/.config/bioamla/config.toml); BIOAMLA_* environment
    variables override both, and a flag on the command line overrides everything.
    """


def _config_target(ctx: click.Context, user: bool):
    """The settings file this command reads or writes: the user's, or the workspace's."""
    from bioamla.common.settings import user_config_path

    if user:
        return user_config_path()
    return _current(ctx.parent.parent).config_path


def _effective(ctx: click.Context, user: bool = False) -> dict[str, dict]:
    """Every setting in effect for this command's workspace, or with ``user``, only
    those in the user file."""
    from bioamla.cli.settings_map import active_layers, effective_settings
    from bioamla.common.settings import read_settings_file, user_config_path

    start = (ctx.parent.parent.obj or {}).get("start")
    try:
        if user:
            path = user_config_path()
            layers = [(path, read_settings_file(path))] if path.is_file() else []
        else:
            layers = active_layers(start)
        return effective_settings(ctx.find_root().command, layers)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@workspace_config.command("list")
@click.option("--user", is_flag=True, help="Show only the user file's settings")
@click.pass_context
def workspace_config_list(ctx: click.Context, user: bool) -> None:
    """Show every setting in effect here, and the file it comes from."""
    from bioamla.cli.console import print_table
    from bioamla.cli.output import emit_result

    rows = sorted(_effective(ctx, user).values(), key=lambda r: r["key"])
    if emit_result({"settings": rows}):
        return
    if not rows:
        click.echo("No settings. Set one with `bioamla workspace config set KEY VALUE`.")
        return
    print_table("Settings", ["key", "value", "from"],
                [[r["key"], repr(r["value"]), r["source"]] for r in rows])  # fmt: skip


@workspace_config.command("get")
@click.argument("key")
@click.option("--user", is_flag=True, help="Look only in the user file")
@click.pass_context
def workspace_config_get(ctx: click.Context, key: str, user: bool) -> None:
    """Show the value a setting has here, and where it comes from."""
    from bioamla.cli.output import emit_result
    from bioamla.cli.settings_map import canonical_key

    found = _effective(ctx, user).get(canonical_key(ctx.find_root().command, key))
    if found is None:
        where = " in the user file" if user else ""
        raise click.ClickException(f"{key} is not set{where} (the command's own default applies)")
    if emit_result(found):
        return
    click.echo(f"{found['value']!r}  (from {found['source']})")


@workspace_config.command("set", context_settings={"ignore_unknown_options": True})
@click.argument("key")
@click.argument("value")
@click.option("--user", is_flag=True, help="Write the user file instead of the workspace's")
@click.pass_context
def workspace_config_set(ctx: click.Context, key: str, value: str, user: bool) -> None:
    """Set KEY to VALUE, converted by the option's own type (an int option gets an
    int; text keeps exactly what you typed). Options that take several values take a
    TOML list, e.g. '["frog", "toad"]'.

    Comments in the file are not kept.

    \b
    Examples:
        bioamla workspace config set detect.energy.threshold_db -25
        bioamla workspace config set --user models.ast.predict.model_path bioamla/scp-frogs
    """
    from bioamla.cli.console import print_success
    from bioamla.cli.output import emit_result
    from bioamla.cli.settings_map import canonical_key, coerce_value, resolve_key
    from bioamla.common.files import write_text_atomic
    from bioamla.common.settings import (
        dump_toml,
        flatten,
        read_settings_file,
        set_dotted,
        unset_dotted,
    )

    root = ctx.find_root().command
    try:
        _, option = resolve_key(root, key)
        key = canonical_key(root, key)
        path = _config_target(ctx, user)
        settings = read_settings_file(path)
        for name in list(flatten(settings)):  # one spelling per option in the file
            if name != key and canonical_key(root, name) == key:
                unset_dotted(settings, name)
        parsed = coerce_value(option, value)
        set_dotted(settings, key, parsed)
        write_text_atomic(path, dump_toml(settings))
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    long_name = next((o for o in option.opts if o.startswith("--")), option.opts[0])
    if emit_result({"key": key, "value": parsed, "file": str(path), "option": long_name}):
        return
    print_success(f"{key} = {parsed!r}  ({path})")


@workspace_config.command("unset")
@click.argument("key")
@click.option("--user", is_flag=True, help="Change the user file instead of the workspace's")
@click.pass_context
def workspace_config_unset(ctx: click.Context, key: str, user: bool) -> None:
    """Remove KEY, so the next level (or the command's default) applies."""
    from bioamla.cli.console import print_success
    from bioamla.cli.output import emit_result
    from bioamla.cli.settings_map import canonical_key
    from bioamla.common.files import write_text_atomic
    from bioamla.common.settings import dump_toml, flatten, read_settings_file, unset_dotted

    root = ctx.find_root().command
    key = canonical_key(root, key)
    try:
        path = _config_target(ctx, user)
        settings = read_settings_file(path)
        removed = False
        for name in list(flatten(settings)):
            if canonical_key(root, name) == key:
                removed = unset_dotted(settings, name) or removed
        if removed:
            write_text_atomic(path, dump_toml(settings))
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    if emit_result({"key": key, "removed": removed, "file": str(path)}):
        return
    if removed:
        print_success(f"Removed {key} ({path})")
    else:
        click.echo(f"{key} was not set in {path}")
