---
name: bioamla-label-qa
description: Check an annotation set in a bioamla workspace before anyone trusts it - well-formed rows inside their recordings, labels in the vocabulary, no duplicates or agent overlaps, and how well the agent's labels agree with a person's on a held-out subset - and give a pass/fail verdict. Use when asked to QA, validate, audit or sign off annotations, especially ones a first-pass labeling run wrote.
---

# Label QA

Checks an annotation set and says whether it can be used: every check is a measurement
against a rule, and the verdict is thresholds the person sets. You run the steps and
report what they say. You never fix a row yourself, and you never change a label a
person made; what fails goes back to the person with the findings.

## Where it runs

The steps run wherever bioamla commands run. With a remote set (pass
`--remote NAME:WORKSPACE` to every step, or set `BIOAMLA_REMOTE`, or `remote` and
`workspace` in the local workspace's settings), the annotations file, the reference
file and the recordings are the server workspace's, and every output is uploaded there.
The run folder's JSON files are also kept locally, so you can read them without fetching.
Run all three steps from the same machine: the later steps read the run folder's files
from where the check step wrote them.

## Steps

Run from the workspace root, in order. Each prints a JSON summary; pass its `run_dir`
to the next step.

1. **Check**: the rows themselves.

   ```
   python <skill>/scripts/check.py annotations.csv --labels frog,bird
   ```

   Writes `checks.json`: one finding per problem, with the row and the rule it broke.
   Rules: `columns` (the required columns exist, and the band columns come as a pair),
   `value` (times, and a band when a row gives one, are finite numbers in order),
   `file` (the recording exists), `bounds` (the row ends inside the recording and its
   band is below Nyquist), `vocabulary` (the label is in `--labels`, and not empty),
   `duplicate` (same file, times, band and label as another row), `overlap` (an agent
   row overlaps another row in time, the labeling skill's own rule; two people's rows
   may overlap, an agent's shouldn't). Without `--labels` the vocabulary isn't checked.
   If `bioamla` itself fails (the server can't be reached, the token is refused), the
   step stops with `{"error": ...}` rather than counting that as a finding.

2. **Agreement**: against a person's labels, where both exist.

   ```
   python <skill>/scripts/agreement.py <run_dir> --reference reference.csv
   ```

   `reference.csv` is the person's annotations for a subset of the recordings: every row
   with a file, a label and a time span counts as the truth (others are skipped and
   counted). Name held-out recordings on which the person found nothing with
   `--recordings a.wav,b.wav`, so the agent's rows there count as false alarms. On the
   covered recordings, the agent rows of the file the check step read are matched to the
   person's by time overlap (`--min-iou`, default 0.5) and scored: per-label precision,
   recall and F1, a confusion matrix, and the disagreements, misses and false alarms with
   their rows, in `agreement.json`. Skip this step when there is no reference; the gate
   then judges the checks alone.

3. **Gate**: the verdict.

   ```
   python <skill>/scripts/gate.py <run_dir> --max-findings 0 --min-f1 0.8
   ```

   Passes when the check step found no more than `--max-findings` problems and, if the
   agreement step ran, the micro-averaged F1 is at least `--min-f1` (`--min-precision`
   and `--min-recall` too, when given; `--require-agreement` fails a run without the
   agreement step). Exit code 0 is a pass, 2 a fail. Use the thresholds the person gives
   you; don't loosen them to pass.

## Report

Tell the person the verdict and every reason in `gate.json`. On a fail, point them at
`checks.json` for the rows to fix (by row number and rule) and at `agreement.json` for
the labels that disagree, the misses and the false alarms. Don't fix rows or relabel
anything yourself; that's the person's call. If a step prints `{"error": ...}`, report
the error as it is and stop.
