#!/bin/bash
# =============================================================================
# Species Detection Workflow
# =============================================================================
# PURPOSE: Detect and identify species vocalizations in audio recordings using
#          specialized acoustic detection algorithms.
#
# FEATURES DEMONSTRATED:
#   - Band-limited energy detection for frequency-specific sounds
#   - RIBBIT detector for periodic/pulsed calls (frogs, insects)
#   - CWT peak detection for call sequences
#   - Accelerating pattern detection for species with increasing call rates
#   - Batch detection across multiple files
#   - Detection visualization with spectrograms
#
# INPUT: Directory of audio recordings
# OUTPUT: Detection results in CSV/JSON format with timestamps and confidence
# =============================================================================

set -e  # Exit on error

# Configuration
AUDIO_DIR="./scp_small"
OUTPUT_DIR="$./detections"

echo "=== Species Detection Workflow ==="
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Band-limited Energy Detection
# Detect sounds within a specific frequency band (e.g., frog calls 500-3000Hz)
echo "Step 1: Running band-limited energy detection..."
bioamla detect energy "$AUDIO_DIR" \
    --low-freq 500 \
    --high-freq 3000 \
    --threshold 0.5 \
    --min-duration 0.1 \
    --output "$OUTPUT_DIR/energy_detections.csv" \
    --format csv

# Step 2: RIBBIT Detection
# Detect periodic pulsed calls characteristic of many frog species
# Pulse rate ~10 pulses/second with 20% tolerance
echo ""
echo "Step 2: Running RIBBIT detector for periodic calls..."
bioamla detect ribbit "$AUDIO_DIR" \
    --pulse-rate 10 \
    --tolerance 0.2 \
    --low-freq 500 \
    --high-freq 4000 \
    --window 2.0 \
    --min-score 0.3 \
    --output "$OUTPUT_DIR/ribbit_detections.csv" \
    --format csv

# Step 3: CWT Peak Detection
# Detect call sequences using continuous wavelet transform
# Good for species with distinct peak patterns in their calls
echo ""
echo "Step 3: Running CWT peak sequence detection..."
bioamla detect peaks "$AUDIO_DIR" \
    --snr 10 \
    --min-distance 0.05 \
    --low-freq 1000 \
    --high-freq 5000 \
    --sequences \
    --min-peaks 3 \
    --output "$OUTPUT_DIR/peak_detections.csv" \
    --format csv

# Step 4: Accelerating Pattern Detection
# Detect species with calls that increase in rate over time
# Common in some frog species during advertisement calls
echo ""
echo "Step 4: Running accelerating pattern detection..."
bioamla detect accelerating "$AUDIO_DIR" \
    --min-pulses 3 \
    --acceleration 1.2 \
    --low-freq 500 \
    --high-freq 3000 \
    --output "$OUTPUT_DIR/accelerating_detections.csv" \
    --format csv

# Step 5: Batch Detection with Multiple Detectors
# Run all detectors on the dataset and merge results
echo ""
echo "Step 5: Running batch detection..."
bioamla detect batch "$AUDIO_DIR" \
    --detector energy \
    --output-dir "$OUTPUT_DIR/batch_energy" \
    --low-freq 500 \
    --high-freq 5000

bioamla detect batch "$AUDIO_DIR" \
    --detector ribbit \
    --output-dir "$OUTPUT_DIR/batch_ribbit" \
    --low-freq 500 \
    --high-freq 4000

# Step 6: Visualize detections on spectrograms
echo ""
echo "Step 6: Generating spectrograms for detected segments..."

# Generate spectrograms for files with detections
for audio_file in "$AUDIO_DIR"/*.wav; do
    if [ -f "$audio_file" ]; then
        filename=$(basename "$audio_file" .wav)
        bioamla audio visualize "$audio_file" \
            --output "$OUTPUT_DIR/spectrograms/${filename}_spectrogram.png" \
            --type mel \
            --n-fft 2048 \
            --hop-length 512 \
            --colormap viridis
    fi
done

echo ""
echo "=== Species Detection Complete ==="
echo "Detection results saved to: $OUTPUT_DIR/"
echo ""
echo "Detection files:"
echo "  - energy_detections.csv: Band-limited energy detections"
echo "  - ribbit_detections.csv: Periodic call detections"
echo "  - peak_detections.csv: CWT peak sequence detections"
echo "  - accelerating_detections.csv: Accelerating pattern detections"
echo "  - spectrograms/: Visual spectrograms for verification"
