---
name: bioamla-first-pass-labeling
description: Do a first pass of labeling a folder of recordings in a bioamla workspace - detect sound events, classify them with an AST model, and write confident ones as annotations while sending the rest to a person for review. Use when asked to pre-label, auto-label or triage recordings.
---

# First-pass labeling

Turns a folder of recordings into draft annotations for a person to check. Every
decision is a threshold on an objective measurement; you run the steps and report what
they say. You never decide a label yourself, and you never change a label a person made.

## Steps

Run from the workspace root, in order. Each prints a JSON summary; pass its `run_dir`
to the next step.

1. **Candidates**: detect events and classify them.

   ```
   python <skill>/scripts/candidates.py <folder> --model <model> --labels <a,b,...>
   ```

   Options: `--low-freq`/`--high-freq` (the detector's band, Hz), `--threshold-db`,
   `--min-duration`, `--segment-seconds` (the model's window).

2. **Evidence**: measure each candidate.

   ```
   python <skill>/scripts/evidence.py <run_dir>
   ```

   Writes `evidence.json`: signal-to-noise ratio in the detector's band (dB), detector
   score, model confidence, band and duration. `margin` is null (the model step gives
   only its top label).

3. **Crops**: a clip and a spectrogram image around each candidate, for review.

   ```
   python <skill>/scripts/crops.py <run_dir> --pad 0.5
   ```

   Writes `<run_dir>/crops/<id>.wav` and `.png`, and `crops.json` with each event's box
   in the clip (seconds and Hz). The box isn't drawn on the image.

4. **Annotate**: accept the candidates that clear every threshold and merge them into
   the annotations CSV.

   ```
   python <skill>/scripts/annotate.py <run_dir> --annotations annotations.csv \
       --min-confidence 0.8 --min-snr 10
   ```

   A candidate is accepted when its label is in the vocabulary and its model
   confidence, SNR and detector score (`--min-score`) are all at or above the
   thresholds. Use the thresholds the person gives you; don't loosen them to accept
   more. Rows a person made (`labeled_by` empty or anything but `agent`) are never
   changed, and an accepted candidate that overlaps one is skipped. Agent rows from an
   earlier run on the same recordings are replaced. A person who corrects an agent row
   should set its `labeled_by` to their name, so later runs keep it. Every decision
   and its reasons go to `decisions.json`.

5. **Review queue**: everything held back, for a person.

   ```
   python <skill>/scripts/review.py <run_dir>
   ```

   Writes `review_queue.json`: each held-back candidate with its reasons, evidence,
   suggested label, clip and image. Don't relabel these yourself; they're for the person.

## Report

Tell the person how many candidates were found, how many were written as annotations,
and how many were held back and why (the review step's `reasons` counts), and point
them at `review_queue.json`. If a step
prints `{"error": ...}`, report the error as it is and stop.
