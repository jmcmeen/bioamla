# Running the server

`bioamla serve` runs bioamla's HTTP API on the machine that holds your recordings, so a
laptop, an agent or [MagPy](https://github.com/jmcmeen/magpy) can work in workspaces there
without copying the data. This page covers installing it, starting it, keeping it running,
and reaching it safely. The design behind it is in
[Workspaces, remote work and agent skills](architecture/workspaces-and-remote.md).

!!! note "What works today"
    The server currently answers `/v1/health` (no token), `/v1/version` and its OpenAPI
    spec at `/v1/openapi.json`. Running operations over HTTP, jobs with progress and cancel,
    the workspace and files routes, and the client and `--remote` routing are being added
    for 0.4.0 (see the [0.4.0 milestone](https://github.com/jmcmeen/bioamla/milestones)).
    The setup on this page stays the same as they land.

## Security model

The server is for **one user on a trusted network**:

- It listens on `127.0.0.1` by default. Reach it through a VPN or an SSH tunnel rather than
  opening it to a network.
- Every request except `/v1/health` must carry the token: `Authorization: Bearer <token>`.
- Every path a request touches must be inside the workspace root (`--root`) or a folder you
  allowed with `--allow-link`. Anything else is refused, including through symlinks.

If you start it on another address, it warns you: anyone who can reach that address and has
the token can run work on the machine.

## Install

The server is an optional extra:

```console
$ pip install 'bioamla[server]'
```

Add the extras for the work you'll run there, e.g. `bioamla[server,all]` for every domain.

## Start it

```console
$ bioamla serve --root ~/bioamla-workspaces
Workspaces: /home/me/bioamla-workspaces
Token: in /home/me/.config/bioamla/server-token
API: http://127.0.0.1:8765/v1/  (spec at /v1/openapi.json)
```

| Option | Default | What it does |
|---|---|---|
| `--root` | (required) | Folder holding the server's workspaces; created if missing |
| `--host` | `127.0.0.1` | Address to listen on |
| `--port` | `8765` | Port to listen on |
| `--token-file` | `~/.config/bioamla/server-token` | File holding the bearer token; created if missing |
| `--allow-link` | (none) | A folder outside `--root` that workspaces may link in place; repeat for several |

Check that it's up:

```console
$ curl http://127.0.0.1:8765/v1/health
$ curl -H "Authorization: Bearer $(cat ~/.config/bioamla/server-token)" \
       http://127.0.0.1:8765/v1/version
```

## The token

On first start, `bioamla serve` writes a random token to `--token-file` and reuses it after
that. To rotate it, stop the server, delete the file and start it again.

- On macOS and Linux the file is created with mode `600` (readable only by you), and a
  token file that others can read is refused.
- On Windows that check isn't made: keep the file in a folder only you can read.
- A token file you write yourself must hold exactly one line, with no spaces in the token.

Clients read the token from an environment variable you choose, never from a stored copy;
see "Remotes" in the [architecture page](architecture/workspaces-and-remote.md).

## Reaching it from another machine

Keep the server on `127.0.0.1` and bring the connection to it.

**SSH tunnel.** From your laptop, forward a local port to the server's:

```console
$ ssh -N -L 8765:127.0.0.1:8765 me@recordings-server.example
```

The API is then at `http://127.0.0.1:8765/v1/` on the laptop, for as long as the tunnel
runs.

**VPN (e.g. Tailscale or WireGuard).** Start the server on the machine's VPN address
instead of localhost, so only devices on your VPN can reach it:

```console
$ bioamla serve --root ~/bioamla-workspaces --host 100.64.0.10
```

The token is still required. Don't use `--host 0.0.0.0` on a machine reachable from the
internet.

## Keep it running with systemd

A **user** service runs as you, with your files and token; enable lingering so it starts at
boot without a login:

```ini
# ~/.config/systemd/user/bioamla.service
[Unit]
Description=bioamla server
After=network-online.target

[Service]
ExecStart=%h/.local/bin/bioamla serve --root %h/bioamla-workspaces --allow-link /srv/recordings
Restart=on-failure

[Install]
WantedBy=default.target
```

```console
$ systemctl --user daemon-reload
$ systemctl --user enable --now bioamla
$ sudo loginctl enable-linger $USER
$ journalctl --user -u bioamla -f
```

Point `ExecStart` at the `bioamla` of the environment you installed it in (for a virtual
environment, `/path/to/venv/bin/bioamla`). For a **system** service, put the unit in
`/etc/systemd/system/`, add `User=` and `Group=` for an unprivileged account, and give
`--root` and `--token-file` paths that account owns.

## Working with recordings already on the machine

Large archives shouldn't be copied into a workspace. Allow their folder with `--allow-link`,
then *link* recordings into a workspace instead of importing them; they're used where they
are:

```console
$ bioamla serve --root /srv/bioamla-workspaces --allow-link /srv/recordings
```

Inside a workspace on that machine:

```console
$ bioamla workspace add /srv/recordings/site-a/2024
```

A linked folder must be under `--root` or one of the `--allow-link` folders; the server
refuses others.
