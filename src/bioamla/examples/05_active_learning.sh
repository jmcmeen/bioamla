#!/bin/bash
# =============================================================================
# Active Learning Annotation Workflow
# =============================================================================
# PURPOSE: Efficiently build labeled datasets by intelligently selecting the
#          most informative samples for human annotation. Reduces annotation
#          effort while maximizing model improvement.
#
# FEATURES DEMONSTRATED:
#   - Initialize active learning session from model predictions
#   - Query samples using uncertainty sampling strategies
#   - Incorporate human annotations iteratively
#   - Track annotation progress and model improvement
#   - Export final labeled dataset
#   - Simulate active learning performance
#
# INPUT: Initial model predictions with confidence scores
# OUTPUT: Optimally-selected samples for annotation, final labeled dataset
# =============================================================================

set -e  # Exit on error

# Configuration
PREDICTIONS_FILE="./predictions.csv"
STATE_FILE="./active_learning_state.json"
OUTPUT_DIR="./active_learning"
BATCH_SIZE=20

echo "=== Active Learning Annotation Workflow ==="
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Initialize active learning session
# The predictions CSV should have columns: file_path, predicted_label, confidence
echo "Step 1: Initializing active learning session..."
bioamla learn init \
    --predictions "$PREDICTIONS_FILE" \
    --output-state "$STATE_FILE" \
    --strategy entropy \
    --diversity-weight 0.3

# Step 2: Query samples for annotation
# Entropy strategy selects samples where the model is most uncertain
echo ""
echo "Step 2: Querying ${BATCH_SIZE} samples for annotation..."
bioamla learn query \
    --state "$STATE_FILE" \
    --n-samples $BATCH_SIZE \
    --output "$OUTPUT_DIR/batch_1_samples.csv"

echo ""
echo ">>> ACTION REQUIRED: Annotate the samples in $OUTPUT_DIR/batch_1_samples.csv"
echo ">>> Add a 'true_label' column with correct labels"
echo ">>> Save as $OUTPUT_DIR/batch_1_annotations.csv"
echo ""

# Step 3: Check active learning status
echo "Step 3: Checking active learning status..."
bioamla learn status --state "$STATE_FILE"

# Step 4: Incorporate annotations (after human annotation)
# This step is typically done after the annotator completes their work
echo ""
echo "Step 4: (Example) Incorporating annotations..."
echo "   Run this after annotation is complete:"
echo "   bioamla learn annotate \\"
echo "       --state \"$STATE_FILE\" \\"
echo "       --annotations \"$OUTPUT_DIR/batch_1_annotations.csv\" \\"
echo "       --annotator \"your_name\""

# Step 5: Query next batch (iterative process)
echo ""
echo "Step 5: (Example) Querying next batch after annotations..."
echo "   bioamla learn query \\"
echo "       --state \"$STATE_FILE\" \\"
echo "       --n-samples $BATCH_SIZE \\"
echo "       --output \"$OUTPUT_DIR/batch_2_samples.csv\""

# Step 6: Export final labeled dataset
echo ""
echo "Step 6: (Example) Exporting labeled dataset..."
echo "   bioamla learn export \\"
echo "       --state \"$STATE_FILE\" \\"
echo "       --output \"$OUTPUT_DIR/labeled_dataset.csv\" \\"
echo "       --format csv"

# Step 7: Simulate active learning to evaluate strategy
# Uses ground truth labels to simulate annotation and measure improvement
echo ""
echo "Step 7: Simulating active learning performance..."
if [ -f "./ground_truth.csv" ]; then
    bioamla learn simulate \
        --predictions "$PREDICTIONS_FILE" \
        --ground-truth "./ground_truth.csv" \
        --n-iterations 10 \
        --samples-per-iteration $BATCH_SIZE \
        --strategy entropy \
        --output "$OUTPUT_DIR/simulation_results.json"
else
    echo "   (Skipped - no ground_truth.csv file found)"
    echo "   To run simulation, provide a CSV with file_path and true_label columns"
fi

echo ""
echo "=== Active Learning Workflow Overview Complete ==="
echo ""
echo "Typical active learning cycle:"
echo "  1. bioamla learn init       - Start session with initial predictions"
echo "  2. bioamla learn query      - Get samples to annotate"
echo "  3. (Human annotates samples)"
echo "  4. bioamla learn annotate   - Add annotations to state"
echo "  5. (Retrain model if desired)"
echo "  6. Repeat steps 2-5 until satisfied"
echo "  7. bioamla learn export     - Export final labeled dataset"
echo ""
echo "Sampling strategies:"
echo "  - entropy: Select samples with highest prediction entropy"
echo "  - least_confidence: Select samples with lowest top confidence"
echo "  - margin: Select samples with smallest margin between top-2 predictions"
echo "  - diversity: Select diverse samples using clustering"
