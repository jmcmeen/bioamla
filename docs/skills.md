# Agent skills

An **agent skill** lets an AI agent (Claude Code, for example) drive bioamla for a task
such as a first pass of labeling. bioamla's skills live in the repository under
`skills/`, one folder per skill.

## Layout

```
skills/<name>/
├── SKILL.md          # what the skill is for, when to use it, how to run it
└── scripts/
    ├── _bioamla.py   # runs `bioamla --json` and reads the result (shared, identical)
    └── <step>.py     # one script per step
```

`SKILL.md` starts with front matter giving the skill's `name` and a `description` that
says when an agent should use it. The body tells the agent which scripts to run, in what
order, and what to report.

## Rules for skill scripts

- **Drive bioamla through its command line.** Scripts run `bioamla --json ...` and act on
  the JSON result; they never import bioamla. That is what lets the same skill run
  against a local install or a remote server: where a command runs is configuration.
- **Decide by thresholds on objective measurements.** Detection scores, model
  confidence, signal-to-noise ratio. The agent doesn't judge; anything below a threshold
  goes to a **review queue** for a person.
- **Never overwrite a person's label.** Rows a person made are kept as they are; agent
  rows are marked `labeled_by=agent`.
- **Paths are relative to the workspace root**, and each run writes into its own folder,
  `runs/<skill>/<timestamp>/`.
- **Files belong to the workspace the commands run in.** With a remote set
  (`BIOAMLA_REMOTE`, a workspace's `remote` setting, or a script's `--remote`), bioamla
  commands run on the server, so a script's own files go there too: `_bioamla.py`'s
  `save_json()` and `publish()` upload what a step writes, and `fetch()` brings a server
  file down to merge into (`bioamla remote current` says which remote is in effect).
- **Print a JSON summary** on standard output (and `{"error": ...}` with a non-zero exit
  when a bioamla command fails), so the agent reads results, not prose.
- **Standard library only.** A script needs Python and `bioamla` on the `PATH`
  (`BIOAMLA_BIN` overrides the command, e.g. a virtual environment's `bin/bioamla`).

## Installing a skill

Copy or link the skill's folder into your agent's skills directory. For Claude Code,
that is `.claude/skills/` in a project, or `~/.claude/skills/` for every project:

```console
$ ln -s /path/to/bioamla/skills/inventory ~/.claude/skills/bioamla-inventory
```

Each skill folder is self-contained (it carries its own copy of `_bioamla.py`), so it can
be installed on its own.

## Evaluating a skill

Every skill has a pass/fail eval in `tests/skills/`. The harness runs the skill's scripts
the way an agent does, as separate processes in a synthetic workspace with known
content, and checks their JSON and the files they wrote. Evals run in CI with the rest of
the integration tests (`pytest -m integration tests/skills`).

## Skills

| Skill | What it does |
|---|---|
| `inventory` | Lists a folder's recordings and flags files that are too short or unreadable. A minimal example of the convention. |
| `label` | A first pass of labeling: finds events with an energy detector, classifies them with an AST model, measures each (SNR, scores), cuts a clip and spectrogram, writes the ones that clear the thresholds as `labeled_by=agent` annotations without touching a person's rows, and queues the rest for review. Its eval runs all five steps on recordings with known events. |
