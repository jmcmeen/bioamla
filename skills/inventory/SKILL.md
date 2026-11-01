---
name: bioamla-inventory
description: Inventory the recordings in a bioamla workspace folder and flag files that are too short or unreadable. Use when asked what recordings a workspace holds or which ones need a look before analysis.
---

# Recording inventory

A minimal example of a bioamla skill (see the "Agent skills" page of the bioamla
docs).

## Run

From the workspace root:

```
python <this skill>/scripts/inventory.py <folder> [--min-duration SECONDS]
```

It prints a JSON summary: `run_dir` (where `inventory.json` was written, under
`runs/inventory/`), the number of files, and `review`, the files a person should look
at, each with its reason.

## Rules

- Report the `review` list as it is. Don't decide yourself whether a flagged file is
  fine; the thresholds decide, and a person reviews.
- Paths are relative to the workspace root.
