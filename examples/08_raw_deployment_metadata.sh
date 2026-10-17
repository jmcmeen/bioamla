#!/usr/bin/env bash
#
# 08 — Describe a raw recorder deployment from its WAV metadata
#
# Field recorders (Wildlife Acoustics Song Meter, AudioMoth, ...) stamp each WAV
# with when and where it was recorded and by which unit. This workflow turns a
# directory of raw recordings into a validated, self-describing dataset without
# decoding any audio and without FFmpeg or a GPU:
#
#   1. pre-flight: every WAV is well-formed PCM (exit 1 otherwise)
#   2. extract recorder metadata into recorder_metadata.csv
#   3. build a dataset metadata.csv (one row per file, label = site folder)
#   4. fill the recorder columns and write dataset.json with the recording
#      span, recorders and sites
#
# Needs only local WAVs. Set RAW_DIR to the deployment root; files are expected
# under one folder per site (e.g. raw/wetlands_1/..., raw/wetlands_2/...).

set -euo pipefail

RAW_DIR="${RAW_DIR:-./raw}"
OUT="./out/08_raw_deployment_metadata"
DATASET="$OUT/dataset"

mkdir -p "$OUT" "$DATASET"

# --- 1. Pre-flight ---------------------------------------------------------
# Exits 1 and lists the damaged files if any WAV fails to parse. Non-WAV files
# (MP3 copies, logs) are reported as skipped, not failed.
bioamla batch audio validate --input-dir "$RAW_DIR" --recursive --output-dir "$OUT"

# --- 2. Recorder metadata per file ------------------------------------------
# recorded_at, latitude, longitude, device_make/model/id, gain, temperature_c,
# battery_voltage, metadata_sources — one row per WAV.
bioamla batch audio metadata --input-dir "$RAW_DIR" --recursive --output-dir "$OUT"
head -3 "$OUT/recorder_metadata.csv"

# --- 3. A dataset manifest: one row per file, labelled by site folder --------
# file_name is relative to the dataset dir; here the dataset simply points at
# the raw files in place (no copies).
{
  echo "file_name,label"
  find "$RAW_DIR" -type f \( -iname '*.wav' -o -iname '*.wave' \) | sort | while read -r f; do
    rel=$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$f" "$DATASET")
    site=$(basename "$(dirname "$f")")
    echo "$rel,$site"
  done
} > "$DATASET/metadata.csv"

# --- 4. Fill the recorder columns, then summarize them in dataset.json ------
bioamla batch audio metadata --input-file "$DATASET/metadata.csv"
bioamla dataset manifest "$DATASET" --name "$(basename "$RAW_DIR")" --kind partitioned

# dataset.json now carries recorded_from/recorded_to, devices and sites (0.01 deg).
python3 - "$DATASET/dataset.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print("recorded:", m["recorded_from"], "->", m["recorded_to"])
print("recorders:", [d["device_id"] for d in m["devices"]])
print("sites:", [(s["latitude"], s["longitude"], s["files"]) for s in m["sites"]])
PY
