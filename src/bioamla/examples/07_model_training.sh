#!/bin/bash
# =============================================================================
# Model Training and Evaluation Workflow
# =============================================================================
# PURPOSE: Train, fine-tune, and evaluate machine learning models for audio
#          classification tasks.
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
# INPUT: Labeled audio dataset with metadata.csv
# OUTPUT: Trained model, evaluation metrics, predictions
# =============================================================================

set -e  # Exit on error

# Configuration
DATASET_DIR="./training_data"
OUTPUT_DIR="./trained_models"
MODEL_NAME="species_classifier"

echo "=== Model Training and Evaluation Workflow ==="
echo ""

# Check system capabilities
bioamla version
bioamla devices

mkdir -p "$OUTPUT_DIR"

# Step 1: Augment training data
# Create variations of training samples to improve model robustness
echo ""
echo "Step 1: Augmenting training data..."
bioamla dataset augment "$DATASET_DIR" \
    --output "$OUTPUT_DIR/augmented_data" \
    --noise 0.3 \
    --time-stretch "0.8-1.2" \
    --pitch-shift "-2-2" \
    --gain "-6-6" \
    --multiplier 3

# Step 2: Fine-tune AST model
# Transfer learning from pretrained AudioSet model
echo ""
echo "Step 2: Fine-tuning AST model..."
bioamla models train ast \
    --training-dir "$OUTPUT_DIR/ast_model" \
    --train-dataset "$OUTPUT_DIR/augmented_data" \
    --num-train-epochs 25 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --warmup-ratio 0.1 \
    --fp16 \
    --dataloader-num-workers 4 \
    --save-strategy epoch \
    --evaluation-strategy epoch \
    --load-best-model-at-end

# Step 3: Train custom CNN model
# Alternative: train a lightweight CNN from scratch
echo ""
echo "Step 3: Training custom CNN model..."
bioamla models train cnn \
    --data-dir "$DATASET_DIR" \
    --output "$OUTPUT_DIR/cnn_model" \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001

# Step 4: Train spectrogram-based classifier
echo ""
echo "Step 4: Training spectrogram classifier..."
bioamla models train spec \
    --data-dir "$DATASET_DIR" \
    --output "$OUTPUT_DIR/spec_model" \
    --model resnet18 \
    --epochs 30 \
    --batch-size 16 \
    --image-size 224 \
    --learning-rate 0.0001

# Step 5: Evaluate models
echo ""
echo "Step 5: Evaluating AST model..."
bioamla models evaluate ast \
    --model-path "$OUTPUT_DIR/ast_model/best_model" \
    --test-dataset "$DATASET_DIR/test" \
    --output "$OUTPUT_DIR/ast_evaluation.json" \
    --per-class-metrics \
    --confusion-matrix "$OUTPUT_DIR/ast_confusion.png"

# Step 6: Create ensemble from multiple models
echo ""
echo "Step 6: Creating model ensemble..."
bioamla models ensemble \
    --model-dirs "$OUTPUT_DIR/ast_model/best_model" "$OUTPUT_DIR/cnn_model" \
    --output "$OUTPUT_DIR/ensemble_predictions.csv" \
    --strategy voting \
    --weights 0.7 0.3

# Step 7: Get model information
echo ""
echo "Step 7: Model information..."
bioamla models info "$OUTPUT_DIR/ast_model/best_model" --model-type ast

# Step 8: List available models
echo ""
echo "Step 8: Available models..."
bioamla models list

# Step 9: Convert model format (optional)
echo ""
echo "Step 9: Converting model format..."
bioamla models convert \
    --input-path "$OUTPUT_DIR/ast_model/best_model" \
    --output-path "$OUTPUT_DIR/ast_model_onnx" \
    --output-format onnx \
    --model-type ast

# Step 10: Push to HuggingFace Hub (optional)
echo ""
echo "Step 10: (Example) Pushing model to HuggingFace Hub..."
echo "   bioamla services hf push-model \\"
echo "       --model-path \"$OUTPUT_DIR/ast_model/best_model\" \\"
echo "       --repo-name \"your-username/$MODEL_NAME\" \\"
echo "       --private"

echo ""
echo "=== Model Training Complete ==="
echo "Models saved to: $OUTPUT_DIR/"
echo ""
echo "Output files:"
echo "  - ast_model/: Fine-tuned AST model"
echo "  - cnn_model/: Custom CNN model"
echo "  - spec_model/: Spectrogram classifier"
echo "  - ast_evaluation.json: Evaluation metrics"
echo "  - ast_confusion.png: Confusion matrix"
echo "  - ensemble_predictions.csv: Ensemble results"
echo ""
echo "Training tips:"
echo "  - Use --fp16 for faster training on supported GPUs"
echo "  - Increase --gradient-accumulation-steps if batch size is limited"
echo "  - Use --load-best-model-at-end to keep the best checkpoint"
echo "  - Monitor with --mlflow-experiment-name for experiment tracking"
