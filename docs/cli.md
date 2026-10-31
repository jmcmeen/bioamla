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
Commands not yet converted report `"result": null`. The older per-command `--json`
flags (`annotation summary`, `audio metadata`, `dataset stats`) still print bare
JSON but are deprecated.

## Commands

::: mkdocs-click
    :module: bioamla.cli.cli
    :command: cli
    :prog_name: bioamla
    :depth: 1
