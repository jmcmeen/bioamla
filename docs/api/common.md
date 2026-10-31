# bioamla.common

Dependency-free building blocks shared by the domain packages. Nothing here
imports a heavy dependency at module load time.

::: bioamla.common
    options:
      members: false

## Constants

::: bioamla.common.constants
    options:
      show_root_heading: false

## Configuration

Config objects are validated on construction, and each has a JSON Schema. The
schemas for this release are published alongside these docs, one file per config
in [`schemas/`](../schemas/ast-train-file.schema.json) (for example
`ast-train-file.schema.json` for the `ast train --config` file). Regenerate them
with `bioamla system schema --all -o docs/schemas`; `bioamla system schema` lists
the names.

To validate a training config in an editor, point its TOML schema support at the
file. With [Taplo](https://taplo.tamasfe.dev/) (Even Better TOML in VS Code), put
this on the first line:

```toml
#:schema https://raw.githubusercontent.com/jmcmeen/bioamla/main/docs/schemas/ast-train-file.schema.json
```

::: bioamla.common.config
    options:
      show_root_heading: false

## Deprecated names

::: bioamla.common.deprecation
    options:
      show_root_heading: false

## Logging

Library records about files, batches and training runs carry structured fields
(`operation`, `file`, `index`, `total`, `run_id`, `epoch`, `elapsed_s`, `status`,
`error`). Per-item and per-step records are logged at DEBUG; a training run's start
and end at INFO. From the command line, `bioamla --log-format json --log-level debug
<command>` writes them to stderr as JSON lines.

::: bioamla.common.log
    options:
      show_root_heading: false

## Files

::: bioamla.common.files
    options:
      show_root_heading: false

## HTTP

::: bioamla.common.http
    options:
      show_root_heading: false

## Batch progress

::: bioamla.common.progress
    options:
      show_root_heading: false
