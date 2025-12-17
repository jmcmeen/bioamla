#!/bin/bash
# =============================================================================
# Visualization Workflow
# =============================================================================
# PURPOSE: Generate spectrograms and visual representations of audio data
#          for analysis, publication, and verification of detections.
#
# FEATURES DEMONSTRATED:
#   - Multiple spectrogram types (mel, STFT, MFCC, waveform)
#   - Customizable visualization parameters
#   - Batch spectrogram generation
#   - Different colormaps and scaling options
#
# INPUT: Audio files or directory of recordings
# OUTPUT: PNG/JPG spectrogram images
# =============================================================================

set -e  # Exit on error

# Configuration
AUDIO_FILE="./recording.wav"
AUDIO_DIR="./recordings"
OUTPUT_DIR="./visualizations"

echo "=== Visualization Workflow ==="
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Generate Mel spectrogram (most common for ML)
echo "Step 1: Generating Mel spectrogram..."
bioamla audio visualize "$AUDIO_FILE" \
    --output "$OUTPUT_DIR/mel_spectrogram.png" \
    --type mel \
    --n-fft 2048 \
    --hop-length 512 \
    --n-mels 128 \
    --fmin 0 \
    --fmax 11025 \
    --colormap viridis \
    --db-scale \
    --title "Mel Spectrogram"

# Step 2: Generate STFT spectrogram (linear frequency scale)
echo ""
echo "Step 2: Generating STFT spectrogram..."
bioamla audio visualize "$AUDIO_FILE" \
    --output "$OUTPUT_DIR/stft_spectrogram.png" \
    --type stft \
    --n-fft 2048 \
    --hop-length 512 \
    --colormap magma \
    --db-scale \
    --title "STFT Spectrogram"

# Step 3: Generate MFCC visualization
echo ""
echo "Step 3: Generating MFCC visualization..."
bioamla audio visualize "$AUDIO_FILE" \
    --output "$OUTPUT_DIR/mfcc.png" \
    --type mfcc \
    --n-mfcc 20 \
    --n-fft 2048 \
    --hop-length 512 \
    --colormap coolwarm \
    --title "MFCC Features"

# Step 4: Generate waveform plot
echo ""
echo "Step 4: Generating waveform visualization..."
bioamla audio visualize "$AUDIO_FILE" \
    --output "$OUTPUT_DIR/waveform.png" \
    --type waveform \
    --title "Audio Waveform"

# Step 5: Customize spectrogram appearance
echo ""
echo "Step 5: Generating customized spectrogram..."
bioamla audio visualize "$AUDIO_FILE" \
    --output "$OUTPUT_DIR/custom_spectrogram.png" \
    --type mel \
    --n-fft 4096 \
    --hop-length 256 \
    --n-mels 256 \
    --fmin 500 \
    --fmax 8000 \
    --colormap plasma \
    --db-scale \
    --figsize 12 4 \
    --dpi 150 \
    --title "High-Resolution Spectrogram (500-8000 Hz)"

# Step 6: Generate spectrograms with different colormaps
echo ""
echo "Step 6: Comparing colormaps..."
for cmap in viridis plasma inferno magma cividis; do
    bioamla audio visualize "$AUDIO_FILE" \
        --output "$OUTPUT_DIR/colormap_${cmap}.png" \
        --type mel \
        --colormap "$cmap" \
        --db-scale \
        --title "Mel Spectrogram ($cmap)"
done

# Step 7: Batch generate spectrograms for directory
echo ""
echo "Step 7: Batch generating spectrograms..."
if [ -d "$AUDIO_DIR" ]; then
    for audio_file in "$AUDIO_DIR"/*.wav; do
        if [ -f "$audio_file" ]; then
            filename=$(basename "$audio_file" .wav)
            bioamla audio visualize "$audio_file" \
                --output "$OUTPUT_DIR/batch/${filename}.png" \
                --type mel \
                --colormap viridis \
                --db-scale
        fi
    done
else
    echo "   (Skipped - $AUDIO_DIR not found)"
fi

# Step 8: Generate zoomed spectrogram (specific time range)
echo ""
echo "Step 8: Generating time-windowed spectrogram..."
bioamla audio trim "$AUDIO_FILE" \
    --output "$OUTPUT_DIR/segment.wav" \
    --start 1.0 \
    --end 3.0

bioamla audio visualize "$OUTPUT_DIR/segment.wav" \
    --output "$OUTPUT_DIR/zoomed_spectrogram.png" \
    --type mel \
    --colormap viridis \
    --db-scale \
    --title "Spectrogram (1-3 seconds)"

echo ""
echo "=== Visualization Complete ==="
echo "Images saved to: $OUTPUT_DIR/"
echo ""
echo "Generated visualizations:"
echo "  - mel_spectrogram.png: Standard Mel spectrogram"
echo "  - stft_spectrogram.png: Linear frequency STFT"
echo "  - mfcc.png: MFCC feature visualization"
echo "  - waveform.png: Time-domain waveform"
echo "  - custom_spectrogram.png: High-resolution custom range"
echo "  - colormap_*.png: Different colormap comparisons"
echo "  - zoomed_spectrogram.png: Time-windowed view"
echo ""
echo "Visualization tips:"
echo "  - Use Mel scale for general analysis and ML"
echo "  - Use STFT for precise frequency measurements"
echo "  - Use MFCC for speech-like sound analysis"
echo "  - Adjust n-fft for time/frequency resolution tradeoff"
echo "  - Use fmin/fmax to focus on frequency bands of interest"
