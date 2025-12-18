#!/bin/bash
# =============================================================================
# BirdSet Audio Spectrogram Transformer Training
# =============================================================================
# PURPOSE: Fine-tune an Audio Spectrogram Transformer (AST) model on BirdSet,
#          the large-scale benchmark dataset for avian bioacoustics.
#
# DATASET: DBD-research-group/BirdSet (HuggingFace)
#   - 6,800+ hours of bird recordings
#   - ~10,000 bird species classes
#   - 8 evaluation subsets for different use cases
#   - License: Various (see dataset card)
#
# AVAILABLE SUBSETS:
#   - HSN: High SNR recordings (cleaner audio)
#   - SSW: Soundscape recordings (more challenging)
#   - UHH: University of Hamburg subset
#   - NBP: North American Bird Project
#   - POW: PowerSet (multi-label)
#   - PER: Peru recordings
#   - NES: Nesting recordings
#   - SNE: Sierra Nevada recordings
#
# MODEL: MIT/ast-finetuned-audioset-10-10-0.4593 (base model)
# OUTPUT: Fine-tuned model saved to ./birdset_model/
#
# REQUIREMENTS: GPU with 16GB+ VRAM recommended
# =============================================================================

set -e

# Configuration
PROJECT_DIR="${PROJECT_DIR:-./my_project}"
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
echo "Training on DBD-research-group/BirdSet ($SUBSET subset)..."
bioamla models train ast \
    --training-dir "$OUTPUT_DIR" \
    --train-dataset "DBD-research-group/BirdSet" \
    --dataset-config "$SUBSET" \
    --num-train-epochs 20 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 3e-5 \
    --warmup-ratio 0.1 \
    --fp16 \
    --dataloader-num-workers 4 \
    --save-strategy epoch \
    --evaluation-strategy epoch \
    --load-best-model-at-end \
    --mlflow-experiment-name "birdset-$SUBSET"

echo ""
echo "=== Training Complete ==="
echo "Model saved to: $OUTPUT_DIR/best_model"
echo ""
echo "To run inference:"
echo "  bioamla models predict ast <audio_file> --model-path $OUTPUT_DIR/best_model"
echo ""
echo "Available BirdSet subsets for training:"
echo "  HSN - High SNR (recommended for starting)"
echo "  SSW - Soundscape recordings"
echo "  UHH - University of Hamburg"
echo "  NBP - North American birds"
echo "  PER - Peru recordings"
echo "  NES - Nesting recordings"
echo "  SNE - Sierra Nevada"
echo ""
echo "Related pre-trained models:"
echo "  - DBD-research-group/Bird-MAE-Base (specialized bird MAE)"
echo "  - DBD-research-group/ConvNeXT-Base-BirdSet-XCL"
