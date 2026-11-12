---
name: dataset
description: Build a training dataset in a bioamla workspace (from catalog downloads or labeled clips, merged and split into train/val/test) and publish it to the Hugging Face Hub only after objective checks pass - licenses that allow it, attribution, redacted recorder locations, no site or recording leaking across splits, and enough examples per class. Use when asked to assemble, check, share, upload or publish a bioacoustic dataset.
allowed-tools: Bash(python3 "${CLAUDE_SKILL_DIR}/scripts/*)
---

# Dataset

Builds a dataset with bioamla's commands, checks it, and publishes it only when the
checks pass. Each check is a measurement against a rule, and the verdict is thresholds
the person sets. You run the steps and report what they say. You don't decide that a
license is fine, a location harmless or a split good enough: the scripts do, and what
fails goes back to the person.

## Where it runs

The steps run wherever bioamla commands run. With a remote set (pass
`--remote NAME:WORKSPACE` to every script, or set `BIOAMLA_REMOTE`, or `remote` and
`workspace` in the local workspace's settings), the dataset is the server workspace's
and the push runs there. The run folder's JSON files are also kept locally, so run the
check, gate and push from the same machine.

## Build

Run from the workspace root. These are plain bioamla commands; use the ones the task
needs, with the person's choices (sources, labels, fractions).

- Download recordings: `bioamla catalogs xc download ...`, `catalogs inat download ...`
  or `catalogs ml download ...` (each writes a `metadata.csv` with a license and attribution per
  file), or extract labeled clips with `bioamla dataset extract-clips`.
- Combine sources: `bioamla dataset merge OUT SRC1 SRC2 ...`.
- Split: `bioamla dataset partition OUT --group-by COLUMN`. Group by what must not be
  both trained and tested on: `source_file` (the default) keeps clips of one recording
  together; a site or device column keeps a whole site or recorder together.
- Credit: `bioamla dataset license OUT --format md` writes `ATTRIBUTIONS.md`.
- Look: `bioamla dataset stats OUT`.

## Check and publish

`${CLAUDE_SKILL_DIR}` is this skill's folder (the one holding this file); in an agent
that doesn't fill it in, use that folder's path. Each script prints a JSON summary; pass
its `run_dir` to the next one.

1. **Check**: what must be settled before publishing.

   ```
   python3 "${CLAUDE_SKILL_DIR}/scripts/check.py" datasets/frogs --license CC-BY-NC-4.0 --group-by site
   ```

   `--license` is the license the person will publish the dataset under. Writes
   `checks.json`: one finding per problem, with the row and the rule it broke.
   Rules: `columns` (`file_name`, `label`, `split` and every `--group-by` column exist),
   `split` (the row has one), `license` (each row's license can be read, isn't
   NoDerivatives, and the dataset's license keeps its conditions: Attribution,
   NonCommercial, ShareAlike with the same license and version), `attribution` (a row
   under an Attribution license names whom to credit), `leakage` (a `--group-by` value,
   by default `source_file`, is in more than one split), and `location`. `--coordinates`
   is how the push will treat per-file locations: `round` (the default, to
   `--coordinate-decimals`, 2 is about 1 km) or `strip` redact them in the metadata and
   the WAV files, and then `location` flags files the push can't vouch for (anything
   but the metadata tables, the manifest, a generated card, audio with no location in
   its metadata, and the license files, as `push-dataset --dry-run` lists them); `keep` publishes them exactly, so `location` flags
   any in the metadata or a sample of the WAV files (and `file` a WAV it can't read).
   It also records the rows per split and label.

2. **Gate**: the verdict.

   ```
   python3 "${CLAUDE_SKILL_DIR}/scripts/gate.py" <run_dir> --min-per-class 5 --max-imbalance 10
   ```

   Passes when the check found no more than `--max-findings` problems (default 0), the
   `--require-splits` have rows (default `train,test`), every label has at least
   `--min-per-class` rows in every split, and, when given, the largest label has at most
   `--max-imbalance` times the rows of the smallest. `--accept RULE` leaves out a rule's
   findings; use it only when the person has said so for that rule (for example
   `--accept location` when they mean to publish exact locations). Exit code 0 is a
   pass, 2 a fail. Use the thresholds the person gives you; don't loosen them to pass.

3. **Push**: only after a pass, and only when the person has confirmed the repository
   and whether it's public.

   ```
   python3 "${CLAUDE_SKILL_DIR}/scripts/push.py" <run_dir> my-org/frog-calls
   ```

   Pushes with the coordinate mode the check assumed and the `--license` it used (written
   to the dataset card, so the Hub shows it), privately unless `--public`.
   Refuses (exit 2) when the gate didn't pass or the dataset's `metadata.csv` changed
   since the check; then check and gate it again. Publishing can't be undone, so never
   push without the person's go-ahead, and never use another way to upload a dataset
   that failed.

## Report

Tell the person the verdict and every reason in `gate.json`. On a fail, point them at
`checks.json` for the rows to fix (by row number and rule). After a push, give them the
repository and any warnings in `push.json`. Don't edit the metadata, relicense files or
drop rows yourself to make a check pass; that's the person's call. If a script prints
`{"error": ...}`, report the error as it is and stop.
