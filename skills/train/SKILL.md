---
name: train
description: Fine-tune an AST audio classifier in a bioamla workspace and decide from metrics whether to keep it - validate the training config, train while recording metrics, evaluate on a fixed evaluation set, and keep or discard the run against a baseline by thresholds the person sets. Use when asked to train, fine-tune, evaluate or compare a bioacoustic model, or to check whether a new model is better.
allowed-tools: Bash(python3 "${CLAUDE_SKILL_DIR}/scripts/*)
---

# Train

Trains a model with bioamla and judges it by numbers. Each step writes JSON into a
`runs/train/<timestamp>/` folder, and the verdict comes from the person's thresholds
applied to the metrics, never from reading training logs. You run the steps and report
what they say. You don't decide that a run "looks good", lower a threshold to keep it, or
change the evaluation set to make a comparison work: what fails goes back to the person.

## Where it runs

The steps run wherever bioamla commands run. With a remote set (pass
`--remote NAME:WORKSPACE` to every script, or set `BIOAMLA_REMOTE`, or `remote` and
`workspace` in the local workspace's settings), training and evaluation run on the
server, on its data and hardware; the config, dataset and evaluation files are paths in
the server's workspace. The run folder's JSON is also kept locally, so run every step
from the same machine.

## Steps

`${CLAUDE_SKILL_DIR}` is this skill's folder (the one holding this file); in an agent
that doesn't fill it in, use that folder's path. Each script prints a JSON summary; pass
its `run_dir` to the next one. Run from the workspace root.

0. **Write the config** with the person's choices, starting from
   `bioamla models ast init-config -o train.toml` (documented, every key optional). Its
   JSON Schema is `bioamla system schema ast-train-file`.

1. **Check the config** for the training data:

   ```
   python3 "${CLAUDE_SKILL_DIR}/scripts/config.py" train.toml --dataset datasets/frogs
   ```

   Checks it exactly as training will (schema, types, augmentation ranges, the dataset,
   the paths), without training. Writes `config.json` with the options the run will use
   and any it would ignore (`warnings`; tell the person). Exit 2: the config is refused,
   with the reason; fix the config and check again.

2. **Train**:

   ```
   python3 "${CLAUDE_SKILL_DIR}/scripts/train.py" <run_dir>
   ```

   Trains with the checked config (refused if it changed since), recording metrics to
   the run's `metrics.jsonl`. Writes `train.json`: `status` (`finished`, `cancelled` or
   `failed`), the final and best `eval_*` metrics, the last training loss and the model's
   path. Exit 2: the run didn't finish; report its `status` and `error`. Training can take
   hours (`--timeout`, default a day), longer than your command may be allowed to wait: run
   it in the background if you can. If the wait ends first, training goes on and nothing is
   recorded as failed: `status` is `running`. Run `train.py` again later to collect the
   result; a run that started is never trained twice.

3. **Evaluate** the run, and the baseline, on the same fixed evaluation set: an audio
   folder and a ground-truth CSV the person chose and that neither model trained on.

   ```
   python3 "${CLAUDE_SKILL_DIR}/scripts/evaluate.py" --run <run_dir> --audio eval/audio --ground-truth eval/labels.csv
   python3 "${CLAUDE_SKILL_DIR}/scripts/evaluate.py" --model bioamla/scp-frogs --audio eval/audio --ground-truth eval/labels.csv
   ```

   The baseline is an earlier run (evaluated the same way) or any model (`--model`, which
   makes its own run folder). Writes `eval.json`: overall and per-class metrics, the
   confusion matrix, and the set's identity (a fingerprint of the audio files' paths and
   contents, its CSV's SHA-256, the files skipped). With a remote the server hashes the
   audio, so nothing is downloaded.

4. **Compare**: the verdict.

   ```
   python3 "${CLAUDE_SKILL_DIR}/scripts/compare.py" <run_dir> --baseline <baseline_run_dir>
   ```

   Keeps the candidate (exit 0) when it finished training, beats the baseline on
   `--metric` (default `f1_score`) by at least `--min-delta` (default 0.01), and no class
   lost more than `--max-class-drop` F1 (default 0.05); otherwise discards it (exit 2).
   Refuses (exit 2) when the two were evaluated on different sets. Use the thresholds
   the person gives you; don't loosen them to keep a run.

## Report

Tell the person the verdict and every reason in `compare.json`, with both values and the
classes that changed most. After a discard, the run and its model stay in its folder;
don't delete them, and don't promote, push or deploy a model yourself. If a script prints
`{"error": ...}`, report the error as it is and stop.
