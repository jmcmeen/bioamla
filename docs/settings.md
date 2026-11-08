# Settings

Any command option can get its default from a settings file or an environment
variable, so you don't have to repeat it on every command line.

## Where settings come from

From lowest to highest precedence:

1. the command's own default
2. your user settings: `~/.config/bioamla/config.toml` (or `$XDG_CONFIG_HOME/bioamla/`;
   `%APPDATA%\bioamla\` on Windows)
3. the current [workspace](architecture/workspaces-and-remote.md)'s `.bioamla/config.toml`
4. environment variables, `BIOAMLA_<COMMAND>_<OPTION>`
5. the flag on the command line

## Writing settings

A table names a command, and its keys name that command's options by their long name
(`-` or `_` both work):

```toml
[detect.energy]
threshold_db = -25
low_freq = 800

[models.ast.predict]
model_path = "bioamla/scp-frogs"
```

Or from the shell:

```console
$ bioamla workspace config set detect.energy.threshold_db -25        # this workspace
$ bioamla workspace config set --user models.ast.predict.model_path bioamla/scp-frogs
$ bioamla workspace config list      # every setting in effect here, and its file
$ bioamla workspace config get detect.energy.threshold_db
$ bioamla workspace config list --user   # only the user file's settings
$ bioamla workspace config unset detect.energy.threshold_db
```

`set` checks that the key names a real option. Keys in a file that don't are
reported as a warning when a command runs.

## Environment variables

The variable for an option is `BIOAMLA_`, then the command path and the option's long
name, upper-cased with `_`:

```console
$ BIOAMLA_DETECT_ENERGY_THRESHOLD_DB=-15 bioamla detect energy rec.wav
```

A few variables aren't options and apply to the library as well as the CLI:

| Variable | Default | What it does |
|---|---|---|
| `BIOAMLA_DECODE_TIMEOUT` | `600` | Seconds FFmpeg may take to decode a file (an MP3, a window of one, a block of a stream) before it's stopped with an error, so a stalled decoder can't hold a command, a server request or a job for ever |
| `BIOAMLA_ENCODE_TIMEOUT` | `600` | Seconds FFmpeg may take to write a file WAV and FLAC can't hold (MP3, M4A, OGG) before it's stopped with an error; nothing half-written is left behind |
| `BIOAMLA_PROGRESS` | | `jsonl` writes progress as JSON lines on stderr (see the CLI page) |
