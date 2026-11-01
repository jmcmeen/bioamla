"""`bioamla remote`: servers to work on, and their workspaces, files and jobs."""

import click

from bioamla.exceptions import BioamlaError


def _client(name: str):
    from bioamla.client import connect

    try:
        return connect(name)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


def _call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e


@click.group()
def remote() -> None:
    """Servers (`bioamla serve`) to work on from this machine.

    A remote is a name, the server's URL and the environment variable holding its
    token, kept in the user's settings file ([remotes.NAME]). The token itself is
    never stored: copy it from the server's token file into the variable.
    """


@remote.command("add")
@click.argument("name")
@click.argument("url")
@click.option(
    "--token-env",
    default=None,
    help="Environment variable holding the token (default: BIOAMLA_<NAME>_TOKEN)",
)
def remote_add(name: str, url: str, token_env: str | None) -> None:
    """Add the server at URL as NAME (or change it).

    \b
    Examples:
        bioamla remote add lab http://lab-server:8765
        export BIOAMLA_LAB_TOKEN=...   # from the server's token file
        bioamla remote ping lab
    """
    from bioamla.cli.console import print_success
    from bioamla.cli.output import emit_result
    from bioamla.client import add_remote

    added = _call(add_remote, name, url, token_env)
    if emit_result(added.to_dict()):
        return
    print_success(f"Added remote {added.name}: {added.url} (token in ${added.token_env})")


@remote.command("list")
def remote_list() -> None:
    """Show the remotes, and whether each one's token is set."""
    import os

    from bioamla.cli.console import print_table
    from bioamla.cli.output import emit_result
    from bioamla.client import remotes

    rows = [{**r.to_dict(), "token_set": bool(os.environ.get(r.token_env))}
            for r in _call(remotes)]  # fmt: skip
    if emit_result({"remotes": rows}):
        return
    if not rows:
        click.echo("No remotes. Add one with `bioamla remote add NAME URL`.")
        return
    print_table("Remotes", ["name", "url", "token"],
                [[r["name"], r["url"], f"${r['token_env']}" + ("" if r["token_set"] else " (not set)")]
                 for r in rows])  # fmt: skip


@remote.command("remove")
@click.argument("name")
def remote_remove(name: str) -> None:
    """Forget the remote NAME (the server is untouched)."""
    from bioamla.cli.console import print_success
    from bioamla.cli.output import emit_result
    from bioamla.client import remove_remote

    removed = _call(remove_remote, name)
    if not removed:
        raise click.ClickException(f"No remote {name!r}")
    if emit_result({"name": name, "removed": True}):
        return
    print_success(f"Removed remote {name}")


@remote.command("ping")
@click.argument("name")
def remote_ping(name: str) -> None:
    """Check that NAME answers and accepts the token; show its versions."""
    from bioamla.cli.console import print_kv
    from bioamla.cli.output import emit_result

    client = _client(name)
    info = _call(client.version)
    if emit_result({"name": name, "url": client.url, **info}):
        return
    print_kv(name, client.url)
    for key in ("bioamla", "api", "workspace_format", "user"):
        print_kv(f"  {key}", info.get(key))


@remote.command("workspaces")
@click.argument("name")
@click.option("--create", "create", default=None, metavar="WS", help="Create workspace WS first")
def remote_workspaces(name: str, create: str | None) -> None:
    """List NAME's workspaces (with --create, make one first)."""
    from bioamla.cli.console import print_success, print_table
    from bioamla.cli.output import emit_result

    client = _client(name)
    if create:
        _call(client.create_workspace, create)
        print_success(f"Created workspace {create} on {name}")
    found = _call(client.workspaces)
    if emit_result({"workspaces": found}):
        return
    if not found:
        click.echo(f"No workspaces on {name}. Create one with --create WS.")
        return
    print_table(f"Workspaces on {name}", ["workspace", "name", "artifacts", "created"],
                [[w["workspace"], w["name"], sum(w.get("artifacts", {}).values()), w["created"]]
                 for w in found])  # fmt: skip


@remote.command("upload")
@click.argument("name")
@click.argument("workspace")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--to", "folder", default="", help="Folder in the workspace (default: its top)")
@click.option("--overwrite", is_flag=True, help="Replace files that exist")
def remote_upload(
    name: str, workspace: str, files: tuple[str, ...], folder: str, overwrite: bool
) -> None:
    """Upload FILES into WORKSPACE on NAME.

    \b
    Example:
        bioamla remote upload lab costa-rica rec1.wav rec2.wav --to raw/site1
    """
    from pathlib import Path

    from bioamla.cli.console import print_success
    from bioamla.cli.output import emit_result

    client = _client(name)
    prefix = folder.strip("/")
    uploaded = []
    for file in files:
        target = f"{prefix}/{Path(file).name}" if prefix else Path(file).name
        uploaded.append(_call(client.upload, workspace, file, target, overwrite))
    if emit_result({"uploaded": uploaded}):
        return
    for item in uploaded:
        print_success(f"{item['path']} ({item['size']} bytes)")


@remote.command("download")
@click.argument("name")
@click.argument("workspace")
@click.argument("paths", nargs=-1, required=True)
@click.option(
    "-o",
    "--output",
    default=".",
    type=click.Path(file_okay=False),
    help="Folder to save into (default: the current one)",
)
def remote_download(name: str, workspace: str, paths: tuple[str, ...], output: str) -> None:
    """Download PATHS (files in WORKSPACE on NAME) into a local folder."""
    from pathlib import Path

    from bioamla.cli.console import print_success
    from bioamla.cli.output import emit_result

    client = _client(name)
    Path(output).mkdir(parents=True, exist_ok=True)
    saved = [str(_call(client.download, workspace, path, output)) for path in paths]
    if emit_result({"downloaded": saved}):
        return
    for path in saved:
        print_success(path)


@remote.command("jobs")
@click.argument("name")
@click.option("--workspace", default=None, help="Only this workspace's jobs")
def remote_jobs(name: str, workspace: str | None) -> None:
    """List NAME's jobs since the server started, newest first."""
    from bioamla.cli.console import print_table
    from bioamla.cli.output import emit_result

    found = _call(_client(name).jobs, workspace)
    if emit_result({"jobs": found}):
        return
    if not found:
        click.echo("No jobs.")
        return

    def progress(job):
        p = job.get("progress")
        return f"{p['done']}/{p['total']}" if p else ""

    print_table(f"Jobs on {name}", ["id", "operation", "workspace", "status", "progress"],
                [[j["id"][:12], j["operation"], j["workspace"], j["status"], progress(j)]
                 for j in found])  # fmt: skip


@remote.command("cancel")
@click.argument("name")
@click.argument("job_id")
def remote_cancel(name: str, job_id: str) -> None:
    """Ask job JOB_ID on NAME to stop (a queued job is dropped)."""
    from bioamla.cli.console import print_success
    from bioamla.cli.output import emit_result

    client = _client(name)
    job = _call(client.cancel, _full_job_id(client, job_id))
    if emit_result(job):
        return
    print_success(f"Job {job['id'][:12]}: {job['status']}")


def _full_job_id(client, job_id: str) -> str:
    """``job_id``, or the one job whose id starts with it (ids are listed shortened)."""
    if len(job_id) >= 32:
        return job_id
    matches = [j["id"] for j in _call(client.jobs) if j["id"].startswith(job_id)]
    if len(matches) != 1:
        raise click.ClickException(
            f"No job {job_id!r}" if not matches else f"{job_id!r} matches several jobs"
        )
    return matches[0]
