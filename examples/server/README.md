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
{"status":"ok"}
```

`WORKSPACES` and `RECORDINGS` (environment variables, or a `.env` file here) choose other
host folders. Recordings are mounted read-only and linked into workspaces in place, never
copied: inside a workspace, `bioamla workspace add /recordings/site-a/2024`.

The default image installs `bioamla[server,cli,audio]`: workspaces, files, audio
windows and signal processing. For prediction and training build with every domain,
which brings torch and is large:

```console
$ docker compose build --build-arg BIOAMLA_EXTRAS=all,cli
```

With a GPU and the NVIDIA container toolkit installed, uncomment the `deploy` block in
`compose.yaml` as well.

## Connect from a laptop

The port is published on `127.0.0.1` only, so another machine reaches it through an SSH
tunnel or a VPN, as the server page describes:

```console
$ ssh -N -L 8765:localhost:8765 lab-host &
$ bioamla remote add lab http://localhost:8765
$ export BIOAMLA_LAB_TOKEN=$(cat bioamla-token)   # the file from the host
$ bioamla remote ping lab
$ bioamla remote workspaces lab --create costa-rica
```

To expose it directly, put a reverse proxy with TLS in front and point it at the
container; don't change the published address to `0.0.0.0`.

## Files

- `Dockerfile`: python slim, ffmpeg, `pip install bioamla[...]`, an unprivileged user,
  `bioamla serve` listening on every interface inside the container.
- `compose.yaml`: the mounts, the secret, the loopback-only port, a health check, and the
  commented-out GPU and user settings.
- `bioamla-token`: yours to create; one line, kept private. Not committed.

The container runs as uid 1000. If the mounted folders belong to another user, uncomment
the `user:` line in `compose.yaml` so files it writes come out owned by you.
