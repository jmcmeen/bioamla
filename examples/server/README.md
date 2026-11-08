# bioamla server in a container

`bioamla serve` under docker compose: the workspace root and the recordings as bind
mounts, the token as a secret, the port published on the host's loopback only. The
[Running the server](https://jmcmeen.github.io/bioamla/server/) page explains the server
itself (the token, reaching it from another machine, linking recordings); this is the
container route to the same thing.

## Start it

From this folder:

```console
$ python3 -c "import secrets; print(secrets.token_hex(32))" > bioamla-token
$ chmod 600 bioamla-token
$ mkdir -p workspaces recordings
$ docker compose up -d --build
$ curl http://127.0.0.1:8765/v1/health
{"bioamla":"0.4.1","ok":true,"result":{"status":"ok"}}
```

`WORKSPACES` and `RECORDINGS` (environment variables, or a `.env` file here) choose other
host folders. Recordings are mounted read-only and linked into workspaces in place, never
copied: inside a workspace, `bioamla workspace add /recordings/site-a/2024`.

The default image installs `bioamla[server,cli,audio]`: workspaces, files, audio
windows and signal processing. For prediction and training build with every domain,
which brings torch and is large, then start the service again so the running container
is replaced by one from the new image:

```console
$ docker compose build --build-arg BIOAMLA_EXTRAS=all,cli
$ docker compose up -d
```

With a GPU and the NVIDIA container toolkit installed, uncomment the `deploy` block in
`compose.yaml` as well.

## Connect from a laptop

The port is published on `127.0.0.1` by default, so another machine reaches it through an
SSH tunnel, as the server page describes:

```console
$ ssh -N -L 8765:localhost:8765 lab-host &
$ bioamla remote add lab http://localhost:8765
$ export BIOAMLA_LAB_TOKEN=$(cat bioamla-token)   # the file from the host
$ bioamla remote ping lab
$ bioamla remote workspaces lab --create costa-rica
```

On a VPN (Tailscale, WireGuard), publish the port on the host's VPN address instead and
laptops on the VPN connect to it directly: `BIND=100.64.0.5 docker compose up -d`, with the
host's own address. Keep `BIND` an address only your machines reach, never `0.0.0.0`; to
expose the server more widely, put a reverse proxy with TLS in front of it.

Docker Engine 28 and newer keep a port published on `127.0.0.1` off the local network.
Older engines let other hosts on the same network segment reach it, so there add a
firewall rule for port 8765 or upgrade Docker.

## Files

- `Dockerfile`: python slim, ffmpeg, `pip install bioamla[...]`, an unprivileged user,
  `bioamla serve` listening on every interface inside the container.
- `compose.yaml`: the mounts, the secret, the published port (`BIND`), a stop grace period
  for running jobs, a health check, and the commented-out GPU and user settings.
- `.dockerignore`: keeps the token and the data folders out of the build context.
- `bioamla-token`: yours to create; one line, kept private. Not committed.

The container runs as uid 1000. If the token or the mounted folders belong to another
user, uncomment the `user:` line in `compose.yaml` and start the service with your ids,
`UID=$(id -u) GID=$(id -g) docker compose up -d` (or write `UID=` and `GID=` lines in
`.env`): the server can then read the token, and files it writes come out owned by you.
