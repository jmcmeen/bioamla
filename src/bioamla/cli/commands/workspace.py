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
    if emit_result(ws.to_dict()):
        return
    print_success(f"Created workspace {ws.name!r}")
    print_kv("Root", ws.root)


@workspace.command("status")
@click.pass_context
def workspace_status(ctx: click.Context) -> None:
    """Show the workspace: name, location and what it holds."""
    from bioamla.cli.console import print_kv
    from bioamla.cli.output import emit_result

    ws = _current(ctx)
    try:
        info = ws.to_dict()
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    if emit_result(info):
        return
    print_kv("Workspace", info["name"])
    print_kv("Root", info["root"])
    print_kv("Created", info["created"])
    counts = info["artifacts"]
    print_kv(
        "Artifacts",
        ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items())) if counts else "none",
    )


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
