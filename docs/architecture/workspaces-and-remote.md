# Workspaces, remote work and agent skills

!!! note "Status: design, planned for 0.4.0"
    This page describes how bioamla is being extended for 0.4.0: project workspaces,
    a server you can run next to your data, a client that sends work to it, and agent
    skills that drive the CLI. Nothing here is released yet. Each piece is tracked as a
    GitHub issue in the [0.4.0 milestone](https://github.com/jmcmeen/bioamla/milestones),
    and this page is the reference those issues point to.

## Why

Bioacoustics work often splits across machines. The recordings sit on a storage server,
training wants a GPU, and the person labeling is on a laptop. bioamla today is a library and a
CLI that run where you are. 0.4.0 lets the same commands, the same agent skills and the
[MagPy](https://github.com/jmcmeen/magpy) desktop app work against data that lives somewhere
else, without a second API to learn or keep in sync.

## Three principles

### 1. The CLI is the operation contract

Any piece of work is a `bioamla` command line. Its outcome is the
[`--json` envelope](../cli.md#machine-readable-output) (`{"bioamla", "command", "ok",
"result" | "error"}`) plus whatever files the command wrote into a workspace.

The CLI, the server, the remote client, MagPy and agent skills all use this one contract. The
server runs commands; it does not reimplement them. Request shapes come from the command
options and the exported [config schemas](../cli.md); response shapes are the envelope.
Typed endpoints may be added later for hot paths, but never as the only way to do something.

### 2. Data lives with its workspace; work goes to the data

A **workspace** is a directory with a `.bioamla/` folder inside:

```
my-survey/
├── .bioamla/
│   ├── workspace.toml      # id, name, format version
│   ├── config.toml         # settings for this workspace (see below)
│   ├── artifacts.json      # linked and imported recordings, datasets, models
│   ├── annotations/        # annotation files, keyed per recording
│   ├── imported/           # copies of imported files
│   ├── datasets/           # datasets built here
│   ├── models/             # models trained or added here
│   ├── runs/               # outputs of jobs and skills, one folder per run
│   └── history.jsonl       # one record per command run in this workspace
└── ...                     # your own files
```

Commands find the workspace by walking up from the current directory, the way git finds a
repository. An **artifact** is either *linked* (referenced by path, never copied, so several
workspaces can share it) or *imported* (copied into `imported/`). Paths inside a workspace are
stored relative to it, so a command line or a skill means the same thing on any machine.

A workspace belongs to one machine. A *remote* workspace lives on the server, under the
server's workspace root, and can link raw recordings where they already are (for example a
large audio archive on the same machine), as long as that location is on the server's
allow-list. Moving work to the data, rather than data to the work, is what makes multi-terabyte
archives practical.

The format is adopted from MagPy's workspace bundle, so MagPy and the CLI open the same
workspaces.

### 3. Location is configuration, not code

Settings cascade, each level overriding the one before:

1. built-in defaults
2. the user file, `~/.config/bioamla/config.toml` (or `$XDG_CONFIG_HOME/bioamla/`)
3. the workspace file, `.bioamla/config.toml`
4. `BIOAMLA_*` environment variables
5. command-line flags

Config supplies command option defaults (through Click's `default_map`). Library functions
keep taking explicit arguments and never read config files themselves.

A **remote** is a named server in user config:

```toml
[remotes.lab]
url = "http://lab-server:8765"
token_env = "BIOAMLA_LAB_TOKEN"   # the token is read from this variable, never stored
```

Choose it per command (`bioamla --remote lab ...`), per shell (`BIOAMLA_REMOTE=lab`), or per
workspace (`remote = "lab"` and `workspace = "my-survey"` in its `config.toml`). With a remote
selected, the client sends the command line to the server as a job, streams its progress and
prints the same envelope or human-readable output a local run would. Output files stay in the
remote workspace until you download them.

## Jobs and history

Every command run inside a workspace appends a record to `history.jsonl`: time, command line,
host, bioamla version, success, duration and output paths. On the server, a **job** is that same
record plus a status, progress events and a cancel operation. So "what has been run on this
data" has one answer, whether the work ran locally or remotely.

Progress comes from the hooks commands already have: in machine mode
(`BIOAMLA_PROGRESS=jsonl`) the progress bar emits one JSON event per update instead of drawing
a bar, and the server forwards those events to clients.

## The server

The server is an optional extra, `pip install bioamla[server]` (FastAPI and uvicorn), started
with:

```console
$ bioamla serve --root /srv/bioamla-workspaces --token-file ~/.config/bioamla/server-token
```

Version 1 of the API (`/v1`, described by a committed OpenAPI spec) covers:

| Area | Endpoints |
|---|---|
| Service | version; the command catalog (every command, option, type and default); config schemas |
| Workspaces | list, create, status, config |
| Files | list, upload, download; later, streamed audio windows for MagPy |
| Jobs | submit a command line, status, progress events (server-sent events), cancel, result envelope, logs |
| History | a workspace's run history |

Each job runs as a `bioamla --json` subprocess with the workspace as its working directory, so
a crash or a cancel can't take the server down, and at most one machine-learning job runs at a
time.

**Security model (first version).** One user on a trusted network: the server binds to
localhost by default and is reached over a VPN or SSH tunnel; every request carries the token;
jobs are command lines only (no shell, no Python); every path a job or request touches must be
inside a workspace root or an allow-listed linked location. The code keeps the seams for a
later small-lab version with several users: every request carries an identity, jobs record it,
and path access goes through a single authorization check.

## Agent skills

Agent skills (a `SKILL.md` plus helper scripts) let an AI agent drive bioamla, for example to
do a first pass of hand labeling. Their rules:

- Scripts call only `bioamla --json` and act on its JSON. Decisions come from thresholds on
  objective measurements (detection scores, model confidence, signal-to-noise), not from the
  agent's judgement.
- Files are referred to by workspace-relative paths, and outputs go to
  `runs/<skill>/<timestamp>/`.
- Anything below threshold goes to a review queue for a person, and an agent never overwrites a
  label a person made.

Because location is configuration, **the same skill runs locally or remotely unchanged**. The
agent runs on your machine and drives the CLI; with a remote selected, the commands, the data
and the compute stay on the server, and only the JSON results and any files you ask for (such as
spectrogram crops for a visual check) come back.

## MagPy

MagPy talks to bioamla through one layer of its code. That layer gains two backends: the
current in-process one, and a remote one built on the bioamla client. "Connect to server" then
lists the server's workspaces, streams audio windows for the spectrogram view and runs heavy
operations as jobs. MagPy will adopt this after the 0.4.x series settles.

## What's in 0.4.0

- Workspaces: the format, `bioamla workspace` commands, cascading config, run history.
- Machine-readable progress and a command catalog.
- The server, the client, `--remote` routing, and pipeline tests that run a remote workflow
  in CI.
- The skills convention and the first-pass labeling skill.
- Removal of names deprecated in 0.3.x, which makes 0.4.0 a minor release.

Later: further skills (label quality checks, dataset building, training and evaluation), a
multi-user server for small labs, and MagPy's remote mode.
