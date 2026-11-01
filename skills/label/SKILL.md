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

## Report

Tell the person how many candidates were found, and where the run folder is. If a step
prints `{"error": ...}`, report the error as it is and stop.
