#!/bin/bash
# =============================================================================
# SCP Frogs Audio Spectrogram Transformer Training
# =============================================================================
# PURPOSE: Fine-tune an Audio Spectrogram Transformer (AST) model on the
#          South Carolina frog species dataset from iNaturalist observations.
#
# DATASET: bioamla/scp-frogs-inat-v1 (HuggingFace)
#   - Frog vocalizations from South Carolina, USA
#   - Sourced from research-grade iNaturalist observations
#   - Multiple species of native frogs and toads
#   - License: See individual observation licenses
#
# ALTERNATIVE DATASET: bioamla/scp-frogs-small
#   - Smaller subset for quick testing and development
#
# MODEL: MIT/ast-finetuned-audioset-10-10-0.4593 (base model)
# OUTPUT: Fine-tuned model saved to ./scp_frogs_model/
#
# REQUIREMENTS: GPU recommended for full training
# =============================================================================

set -e

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"

echo "=== SCP Frogs AST Training Workflow ==="
echo ""

# Check system capabilities
bioamla version
bioamla devices

# Option 1: Train on full dataset (recommended for production)
echo "Training on bioamla/scp-frogs-inat-v1 dataset..."
bioamla models train ast \
    --training-dir "${PROJECT_DIR}/scp_frogs_model" \
    --train-dataset "bioamla/scp-frogs-inat-v1" \
    --num-train-epochs 25 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --fp16 \
    --dataloader-num-workers 4 \
    --save-strategy epoch \
    --eval-strategy epoch \
    --load-best-model-at-end \
    --mlflow-experiment-name "scp-frogs"

echo ""
echo "=== Training Complete ==="
echo "Model saved to: ${PROJECT_DIR}/scp_frogs_model/best_model"
echo ""
echo "To run inference:"
echo "  bioamla models predict ast <audio_file> --model-path ${PROJECT_DIR}/scp_frogs_model/best_model"
echo ""
echo "Or use the pre-trained model from HuggingFace:"
echo "  bioamla models predict ast <audio_file> --model-path bioamla/scp-frogs"
echo ""
echo "For quick testing, use the smaller dataset:"
echo "  --train-dataset bioamla/scp-frogs-small"
