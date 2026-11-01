"""`bioamla serve`: the HTTP API, next to the data."""

import click

from bioamla.exceptions import BioamlaError

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@click.command("serve")
@click.option(
    "--root",
    required=True,
    type=click.Path(file_okay=False),
    help="Folder holding the server's workspaces (created if missing)",
)
@click.option("--host", default="127.0.0.1", show_default=True, help="Address to listen on")
@click.option("--port", default=8765, show_default=True, type=int, help="Port to listen on")
@click.option(
    "--token-file",
    default="~/.config/bioamla/server-token",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="File holding the bearer token (created if missing; mode 600 on macOS and Linux)",
)
@click.option(
    "--allow-link",
    "linkable",
    multiple=True,
    type=click.Path(exists=True, file_okay=False),
    help="A folder outside --root that workspaces may link in place (repeatable)",
)
@click.option(
    "--max-jobs",
    default=2,
    show_default=True,
    type=click.IntRange(min=1),
    help="Jobs that may run at once (machine-learning jobs run one at a time)",
)
def serve(
    root: str, host: str, port: int, token_file: str, linkable: tuple[str, ...], max_jobs: int
) -> None:
    """Run the bioamla HTTP API so other machines can work in this machine's
    workspaces. Needs the [server] extra.

    It's for one user on a trusted network: it listens on localhost unless told
    otherwise; reach it over a VPN or an SSH tunnel. Clients send the token in
    `Authorization: Bearer <token>`.

    \b
    Examples:
        bioamla serve --root ~/bioamla-workspaces
        bioamla serve --root /srv/workspaces --allow-link /srv/recordings
    """
    from pathlib import Path

    from bioamla.cli.console import print_kv, print_warning
    from bioamla.cli.output import json_mode

    if json_mode():
        raise click.ClickException("serve has no --json output (it runs until stopped)")

    try:
        from bioamla._extras import import_extra
        from bioamla.server import ServerConfig, create_app, read_or_create_token

        uvicorn = import_extra("uvicorn", "server")

        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        token = read_or_create_token(token_file)
        config = ServerConfig(
            root=root_path,
            token=token,
            linkable=[Path(p).expanduser() for p in linkable],
            max_jobs=max_jobs,
        )
        app = create_app(config)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    except OSError as e:  # e.g. no permission to create --root or the token file
        raise click.ClickException(f"Can't start the server: {e}") from e

    if host not in _LOCAL_HOSTS:
        print_warning(
            f"Listening on {host}: anyone who can reach this address and has the token can "
            "run commands here. Prefer a VPN or an SSH tunnel to a localhost server."
        )
    print_kv("Workspaces", root_path.resolve())
    print_kv("Token", f"in {Path(token_file).expanduser()}")
    shown = f"[{host}]" if ":" in host else host  # IPv6 addresses need brackets in URLs
    print_kv("API", f"http://{shown}:{port}/v1/  (spec at /v1/openapi.json)")
    uvicorn.run(app, host=host, port=port, log_level="info")
