#!/bin/bash
# =============================================================================
# ESC-50 Audio Spectrogram Transformer Training
# =============================================================================
# PURPOSE: Fine-tune an Audio Spectrogram Transformer (AST) model on the ESC-50
#          environmental sound classification dataset.
#
# DATASET: ashraq/esc50 (HuggingFace)
#   - 2000 5-second environmental audio clips
#   - 50 classes across 5 categories: animals, natural soundscapes,
#     human non-speech, domestic sounds, urban noises
#   - License: CC BY-NC 3.0
#
# MODEL: MIT/ast-finetuned-audioset-10-10-0.4593 (base model)
# OUTPUT: Fine-tuned model saved to ./esc50_model/
#
# REQUIREMENTS: GPU recommended (training takes ~30 min on RTX 3090)
# =============================================================================

set -e

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"

echo "=== ESC-50 AST Training Workflow ==="
echo ""

# Check system
bioamla version
bioamla devices

# Train AST model on ESC-50
# The dataset will be automatically downloaded from HuggingFace
bioamla models train ast \
    --training-dir "${PROJECT_DIR}/esc50_model" \
    --train-dataset "ashraq/esc50" \
    --num-train-epochs 3 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --fp16 \
    --dataloader-num-workers 4 \
    --save-strategy epoch \
    --eval-strategy epoch \
    --load-best-model-at-end \
    --mlflow-experiment-name "esc50-ast"

echo ""
echo "=== Training Complete ==="
echo "Model saved to: ${PROJECT_DIR}/esc50_model/best_model"
echo ""
echo "To run inference with this model:"
echo "  bioamla models predict ast <audio_file> --model-path ${PROJECT_DIR}/esc50_model/best_model"
echo ""
echo "Or use the pre-trained bioamla/ast-esc50 model from HuggingFace:"
echo "  bioamla models predict ast <audio_file> --model-path bioamla/ast-esc50"
