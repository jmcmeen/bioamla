#!/bin/bash
# =============================================================================
# BirdSet Audio Spectrogram Transformer Training
# =============================================================================
# PURPOSE: Fine-tune an Audio Spectrogram Transformer (AST) model on BirdSet,
#          the large-scale benchmark dataset for avian bioacoustics.
#
# DATASET: samuelstevens/BirdSet (HuggingFace - Parquet format)
#   - Large-scale bird sound benchmark
#   - Multiple evaluation subsets
#   - Compatible with datasets 4.x+ (no trust_remote_code needed)
#   - License: Various (see dataset card)
#
# AVAILABLE SUBSETS:
#   - HSN: High SNR recordings (cleaner audio, recommended for starting)
#   - NBP: North American Bird Project
#   - NES: Nesting recordings
#   - PER: Peru recordings
#
# MODEL: MIT/ast-finetuned-audioset-10-10-0.4593 (base model)
# OUTPUT: Fine-tuned model saved to ./birdset_model/
#
# REQUIREMENTS: GPU with 16GB+ VRAM recommended
# =============================================================================

set -e

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"
SUBSET="${1:-HSN}"  # Default to High SNR subset
OUTPUT_DIR="${PROJECT_DIR}/birdset_model_${SUBSET}"

echo "=== BirdSet AST Training Workflow ==="
echo "Subset: $SUBSET"
echo "Output: $OUTPUT_DIR"
echo ""

# Check system
bioamla version
bioamla devices

# Train AST model on BirdSet
# Note: BirdSet uses 32kHz sample rate
# Use "dataset:config" format to specify the subset
echo "Training on samuelstevens/BirdSet ($SUBSET subset)..."
bioamla models train ast \
    --training-dir "$OUTPUT_DIR" \
    --train-dataset "samuelstevens/BirdSet:${SUBSET}" \
    --split "train" \
    --category-label-column "ebird_code" \
    --num-train-epochs 3 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 3e-5 \
    --fp16 \
    --dataloader-num-workers 4 \
    --save-strategy epoch \
    --eval-strategy epoch \
    --load-best-model-at-end \
    --mlflow-experiment-name "birdset-$SUBSET"

echo ""
echo "=== Training Complete ==="
echo "Model saved to: $OUTPUT_DIR/best_model"
echo ""
echo "To run inference:"
echo "  bioamla models predict ast <audio_file> --model-path $OUTPUT_DIR/best_model"
echo ""
echo "Available BirdSet subsets for training (samuelstevens/BirdSet):"
echo "  HSN - High SNR (recommended for starting)"
echo "  NBP - North American birds"
echo "  NES - Nesting recordings"
echo "  PER - Peru recordings"
echo ""
echo "Related pre-trained models:"
echo "  - DBD-research-group/Bird-MAE-Base (specialized bird MAE)"
echo "  - DBD-research-group/ConvNeXT-Base-BirdSet-XCL"
