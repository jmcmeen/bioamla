#!/bin/bash
# =============================================================================
# iNaturalist Dataset Creation and Training Workflow
# =============================================================================
# PURPOSE: Download audio observations from iNaturalist and train a custom
#          species classifier model.
#
# WORKFLOW:
#   1. Download audio from iNaturalist by taxon IDs
#   2. Convert audio to consistent WAV format
#   3. Fine-tune AST model on the collected dataset
#   4. Run inference using a test dataset
#
# EXAMPLE TAXA (South Carolina Frogs):
#   24268  - Anaxyrus terrestris (Southern Toad)
#   65982  - Anaxyrus fowleri (Fowler's Toad)
#   23930  - Lithobates catesbeianus (American Bullfrog)
#   24263  - Lithobates clamitans (Green Frog)
#   65979  - Lithobates sphenocephalus (Southern Leopard Frog)
#   66002  - Pseudacris crucifer (Spring Peeper)
#   66012  - Pseudacris feriarum (Upland Chorus Frog)
#   60341  - Hyla cinerea (Green Treefrog)
#   64968  - Hyla chrysoscelis (Cope's Gray Treefrog)
#   64977  - Hyla squirella (Squirrel Treefrog)
#   24256  - Acris gryllus (Southern Cricket Frog)
#
# PRE-BUILT ALTERNATIVES:
#   - bioamla/scp-frogs-inat-v1: Full SCP frogs dataset (HuggingFace)
#   - bioamla/scp-frogs-small: Small test subset (HuggingFace)
#   - bioamla/scp-frogs: Pre-trained model (HuggingFace)
# =============================================================================

set -e

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"

echo "=== iNaturalist Dataset Workflow ==="
echo ""

# Step 1: Download audio observations from iNaturalist
# These are South Carolina frog species taxon IDs
echo "Step 1: Downloading audio from iNaturalist..."
bioamla services inat download "${PROJECT_DIR}/frogs_dataset" \
    --taxon-ids "24268,65982,23930,24263,65979,66002,66012,60341,64968,64977,24256" \
    --quality-grade research \
    --obs-per-taxon 100

# Step 2: Convert all audio files to WAV format
echo ""
echo "Step 2: Converting audio to WAV format..."
bioamla audio convert "${PROJECT_DIR}/frogs_dataset" wav

# Step 3: Fine-tune an AST model on the downloaded dataset
echo ""
echo "Step 3: Training AST model..."
bioamla models train ast \
    --training-dir "${PROJECT_DIR}/frogs_model" \
    --train-dataset "${PROJECT_DIR}/frogs_dataset" \
    --num-train-epochs 25 \
    --per-device-train-batch-size 8 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --fp16 \
    --dataloader-num-workers 4 \
    --save-strategy epoch \
    --evaluation-strategy epoch \
    --load-best-model-at-end

# Step 4: Test with pre-built dataset from HuggingFace
echo ""
echo "Step 4: Testing with bioamla/scp-frogs-small dataset..."
bioamla models predict ast "bioamla/scp-frogs-small" \
    --batch \
    --model-path "${PROJECT_DIR}/frogs_model/best_model" \
    --output "${PROJECT_DIR}/frogs_predictions.csv" \
    --top-k 5

echo ""
echo "=== Workflow Complete ==="
echo ""
echo "Created files:"
echo "  ${PROJECT_DIR}/frogs_dataset/     - Downloaded iNaturalist audio"
echo "  ${PROJECT_DIR}/frogs_model/       - Trained model directory"
echo "  ${PROJECT_DIR}/frogs_predictions.csv - Test predictions"
echo ""
echo "Alternative: Use pre-built resources from HuggingFace:"
echo "  Dataset: bioamla/scp-frogs-inat-v1"
echo "  Model:   bioamla/scp-frogs"
echo ""
echo "Quick inference with pre-trained model:"
echo "  bioamla models predict ast <audio> --model-path bioamla/scp-frogs"
