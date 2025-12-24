#!/bin/bash
# =============================================================================
# Model Training and Evaluation Workflow
# =============================================================================
# PURPOSE: Train, fine-tune, and evaluate machine learning models for audio
#          classification tasks using publicly available datasets.
#
# FEATURES DEMONSTRATED:
#   - Dataset augmentation for training
#   - Fine-tuning Audio Spectrogram Transformer (AST) models
#   - Training custom CNN classifiers
#   - Model evaluation with metrics
#   - Ensemble prediction strategies
#   - Model format conversion
#   - Pushing models to HuggingFace Hub
#
# AVAILABLE DATASETS (HuggingFace):
#   - ashraq/esc50: Environmental sounds (50 classes, 2000 clips)
#   - bioamla/scp-frogs-inat-v1: Frog species from iNaturalist
#   - bioamla/scp-frogs-small: Small frog dataset for testing
#   - samuelstevens/BirdSet: Bird sounds (subsets: HSN, NBP, NES, PER)
#
# AVAILABLE MODELS (HuggingFace):
#   - MIT/ast-finetuned-audioset-10-10-0.4593: Base AST model
#   - bioamla/ast-esc50: ESC-50 trained model
#   - bioamla/scp-frogs: Frog species classifier
#
# OUTPUT: Trained model, evaluation metrics, predictions
# =============================================================================

set -e  # Exit on error


# Choose your dataset
# Option 1: ESC-50 environmental sounds
DATASET="ashraq/esc50"
# Option 2: Frog species
# DATASET="bioamla/scp-frogs-inat-v1"
# Option 3: Quick test with small dataset
# DATASET="bioamla/scp-frogs-small"

OUTPUT_DIR="./trained_models"
MODEL_NAME="esc50_classifier"

echo "=== Model Training and Evaluation Workflow ==="
echo "Dataset: $DATASET"
echo ""

# Check system capabilities
bioamla version
bioamla devices

mkdir -p "$OUTPUT_DIR"

# Step 1: Fine-tune AST model on HuggingFace dataset
# Transfer learning from pretrained AudioSet model
echo ""
echo "Step 1: Fine-tuning AST model on $DATASET..."
bioamla models train ast \
    --training-dir "$OUTPUT_DIR/ast_model" \
    --train-dataset "$DATASET" \
    --num-train-epochs 3 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --fp16 \
    --dataloader-num-workers 4 \
    --save-strategy epoch \
    --eval-strategy epoch \
    --load-best-model-at-end \
    --mlflow-experiment-name "bioamla-training"

# Step 2: Get model information
echo ""
echo "Step 2: Model information..."
bioamla models info "$OUTPUT_DIR/ast_model/best_model" --model-type ast

# Step 3: List available models
echo ""
echo "Step 3: Available pre-trained models..."
bioamla models list

# Step 4: Convert model format (optional - for deployment)
echo ""
echo "Step 4: Converting model to ONNX format..."
bioamla models convert \
    "$OUTPUT_DIR/ast_model/best_model" \
    "$OUTPUT_DIR/ast_model_onnx" \
    --format onnx \
    --model-type ast

# Step 5: Push to HuggingFace Hub (optional)
echo ""
echo "Step 5: (Example) Pushing model to HuggingFace Hub..."
echo "   bioamla services hf push-model \\"
echo "       \"$OUTPUT_DIR/ast_model/best_model\" \\"
echo "       \"your-username/$MODEL_NAME\" \\"
echo "       --private"

echo ""
echo "=== Model Training Complete ==="
echo "Model saved to: $OUTPUT_DIR/ast_model/best_model"
echo ""
echo "To run inference with your trained model:"
echo "  bioamla models predict ast <audio_file> --model-path $OUTPUT_DIR/ast_model/best_model"
echo ""
echo "Pre-trained alternatives available on HuggingFace:"
echo "  - MIT/ast-finetuned-audioset-10-10-0.4593 (AudioSet, 527 classes)"
echo "  - bioamla/ast-esc50 (ESC-50, 50 environmental sounds)"
echo "  - bioamla/scp-frogs (Frog species classification)"
echo ""
echo "Training tips:"
echo "  - Use --fp16 for faster training on supported GPUs"
echo "  - Increase --gradient-accumulation-steps if batch size is limited"
echo "  - Use --load-best-model-at-end to keep the best checkpoint"
echo "  - Monitor with --mlflow-experiment-name for experiment tracking"
