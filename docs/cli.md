# CLI Reference

The full `bioamla` command tree, generated from the CLI itself (`--help`), so it
always matches the installed version.

## Machine-readable output

`bioamla --json <command> ...` prints exactly one JSON document on stdout. Progress
bars, tables and messages go to stderr, so stdout always parses:

```console
$ bioamla --json annotation summary labels.csv
{"bioamla": "0.3.1", "command": "annotation summary", "ok": true, "result": {"total_annotations": 2, ...}}
```

On failure the command exits with status 1 (2 for a usage error) and prints:

```json
{"bioamla": "0.3.1", "command": "dataset stats", "ok": false,
 "error": {"type": "NotFoundError", "message": "..."}}
```

`error.type` is the bioamla exception name (`NotFoundError`, `InvalidInputError`,
`DependencyError`, ...) or a Click error name (`UsageError`, `MissingParameter`,
`BadParameter`, ...). Some failures also carry `error.details` with structured data,
for example the per-dataset results when `dataset license --batch` fails for some
datasets.
In `result`, paths are strings, NaN and infinity are `null`, and keys are snake_case.
Every command reports a result except `audio play`, which refuses `--json`. Commands
that would prompt (`system deps --install`, `catalogs hf cache --purge`) need `--yes`
with `--json`. The older per-command `--json` flags (`annotation summary`,
`audio metadata`, `dataset stats`) still print bare JSON but are deprecated.

### Progress events

Set `BIOAMLA_PROGRESS=jsonl` and commands with a progress bar write progress as JSON
lines on stderr instead of drawing the bar:

```console
$ BIOAMLA_PROGRESS=jsonl bioamla --json batch audio features --input-dir rec --output-dir out
{"event": "start", "description": "Computing features", "total": null}
{"event": "progress", "description": "Computing features", "done": 1, "total": 10}
...
{"event": "end", "description": "Computing features", "done": 10, "total": 10, "ok": true}
```

A progress event is sent each time the count crosses another whole percent (every item
of a batch of 100 or fewer), at least every 0.1 s otherwise, on a change of description,
and for the final count, never twice for the same state. Events are sent with `--quiet`
too. stderr can also carry messages for people, so read the lines that are
JSON objects with an `"event"` key. With `--json`, stdout is still exactly one envelope.

## Commands

::: mkdocs-click
    :module: bioamla.cli.cli
    :command: cli
    :prog_name: bioamla
    :depth: 1
