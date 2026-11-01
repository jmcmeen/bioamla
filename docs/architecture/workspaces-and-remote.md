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
else, through one set of operations that every front end shares.

## Three principles

### 1. Operations are the contract

Every piece of work bioamla can do is an **operation**: a named unit of work such as
`detect.energy`, `audio.features` or `batch.detect`, with

- a typed **request**: a validated config (so it has a JSON Schema), whose path fields are
  marked as paths;
- a **function** in the operation's domain package that does the whole job, including what a
  command does today beyond calling the library (CSV-mode merging, writing outputs);
- a typed **result**, the same object `--json` reports.

A small registry, `bioamla.operations`, lists them all. The CLI and the server are both thin
front ends to it:

- **The CLI** parses a command's options into the request, runs the operation, and prints the
  result for people or as the [`--json` envelope](../cli.md#machine-readable-output).
- **The server** exposes each operation as a typed endpoint generated from the registry (so its
  OpenAPI spec always matches), running long operations as jobs.

Operations live in their domain packages (`bioamla.detect` declares the detection operations,
and so on); the registry is an index, not a separate layer.

**The library stays the API.** Operations sit on top of the library; they never replace it.
The domain functions (`frame_features`, `batch_detect_files`, ...) remain the public API
anyone can import, with explicit arguments, documentation and exports from each domain's
`__init__`. An operation is a thin adapter: it turns a typed request into calls to public
library functions and wraps the result, adding only what the front ends need (a name, path
checks, progress and cancel). Logic that moves out of a command today (CSV-mode merging,
writing outputs) moves into public library functions first, and the operation calls those. So
anything the CLI or the server can do, a script can do by importing bioamla, and that's tested.

An operation runs with an **operation context** supplied by its front end: where it runs (a
workspace), progress and cancel hooks, and the check every path in the request goes through.
Locally the check allows any path; on the server it confines paths to the allowed folders.

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
stored relative to it, so a request or a skill means the same thing on any machine.

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
selected, the CLI builds the operation request as usual and sends it to the server instead of
running it, streams its progress, and prints the same envelope or human-readable output a local
run would. Output files stay in the remote workspace until you download them.

## Jobs and history

Every command run inside a workspace appends a record to `history.jsonl`: time, command line,
host, bioamla version, success, duration and output paths. On the server, a **job** is an
operation run with that same record, plus a status, progress events and a cancel operation. So "what has been run on this
data" has one answer, whether the work ran locally or remotely.

Progress comes from the operation context's progress hook. On the command line it drives the
progress bar (or, with `BIOAMLA_PROGRESS=jsonl`, JSON progress events); on the server it feeds
the job's progress events, which clients follow.

## The server

The server is an optional extra, `pip install bioamla[server]` (FastAPI and uvicorn), started
with:

```console
$ bioamla serve --root /srv/bioamla-workspaces --token-file ~/.config/bioamla/server-token
```

Version 1 of the API (`/v1`, described by a committed OpenAPI spec) covers:

| Area | Endpoints |
|---|---|
| Service | version; the operation catalog (every operation with its request and result schemas) |
| Operations | one endpoint per operation; quick ones answer directly, long ones start a job |
| Workspaces | list, create, status, config |
| Files | list, upload, download; later, streamed audio windows for MagPy |
| Jobs | status, progress events (server-sent events), cancel, result, logs |
| History | a workspace's run history |

Quick operations (listing files, reading metadata, an audio window) run in the server process.
Long ones (batches, detection over many files, training) run as jobs in worker processes, so
a crash or a cancel can't take the server down, and at most one machine-learning job runs at a
time. The server needs no CLI: it calls the operations directly.

**Security model (first version).** One user on a trusted network: the server binds to
localhost by default and is reached over a VPN or SSH tunnel; every request carries the token;
only registered operations can run, and only those marked as remote-safe (not local playback,
installing system packages or starting another server); every path field of a request must
resolve inside a workspace root or an allow-listed linked location. The code keeps the seams for a
later small-lab version with several users: every request carries an identity, jobs record it,
and path access goes through a single authorization check.

## Agent skills

Agent skills (a `SKILL.md` plus helper scripts) let an AI agent drive bioamla, for example to
do a first pass of hand labeling. Their rules:

- Scripts call `bioamla --json` (that is, operations through the CLI) and act on its JSON. Decisions come from thresholds on
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

MagPy talks to bioamla through one layer of its code. That layer gains two backends: an
in-process one that runs operations directly, and a remote one that calls the same operations on
a server through the bioamla client. "Connect to server" then
lists the server's workspaces, streams audio windows for the spectrogram view and runs heavy
operations as jobs. MagPy will adopt this after the 0.4.x series settles.

## What's in 0.4.0

- Workspaces: the format, `bioamla workspace` commands, cascading config, run history.
- Machine-readable progress and a command catalog.
- The operations registry, with every command converted to run through an operation.
- The server, the client, `--remote` routing, and pipeline tests that run a remote workflow
  in CI.
- The skills convention and the first-pass labeling skill.
- Removal of names deprecated in 0.3.x, which makes 0.4.0 a minor release.

Later: further skills (label quality checks, dataset building, training and evaluation), a
multi-user server for small labs, and MagPy's remote mode.
