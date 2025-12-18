#!/bin/bash
# =============================================================================
# Data Acquisition Workflow
# =============================================================================
# PURPOSE: Acquire audio recordings from online biodiversity databases for
#          training, validation, or research purposes.
#
# FEATURES DEMONSTRATED:
#   - Search and download from iNaturalist
#   - Search and download from Xeno-canto
#   - Search and download from Macaulay Library
#   - Species name lookup and conversion
#   - Dataset organization and metadata management
#   - Cache management
#
# INPUT: Species names, taxonomic IDs, or geographic coordinates
# OUTPUT: Organized audio dataset with metadata
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"
OUTPUT_DIR="${PROJECT_DIR}/acquired_data"
SPECIES_COMMON="American Bullfrog"
SPECIES_SCIENTIFIC="Lithobates catesbeianus"

echo "=== Data Acquisition Workflow ==="
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Species name lookup
# Convert between common and scientific names
echo "Step 1: Looking up species information..."
bioamla services species lookup "$SPECIES_COMMON" --info
bioamla services species lookup "$SPECIES_SCIENTIFIC" --to-common

# Search for related species
echo ""
echo "Searching for related species..."
bioamla services species search "bullfrog" --limit 5

# Step 2: Search iNaturalist for observations
echo ""
echo "Step 2: Searching iNaturalist..."
bioamla services inat search \
    --species "$SPECIES_SCIENTIFIC" \
    --quality-grade research \
    --has-sounds \
    --limit 5

# Step 3: Download audio from iNaturalist
echo ""
echo "Step 3: Downloading from iNaturalist..."
bioamla services inat download "$OUTPUT_DIR/inat" \
    --taxon-name "$SPECIES_SCIENTIFIC" \
    --quality-grade research \
    --obs-per-taxon 5

# Step 4: Search Xeno-canto for bird recordings
echo ""
echo "Step 4: Searching Xeno-canto..."
BIRD_SPECIES="Turdus migratorius"
bioamla services xc search \
    --species "$BIRD_SPECIES" \
    --quality A \
    --max-results 5 \
    --format table

# Step 5: Download from Xeno-canto
echo ""
echo "Step 5: Downloading from Xeno-canto..."
bioamla services xc download \
    --species "$BIRD_SPECIES" \
    --quality A \
    --max-recordings 5 \
    --output-dir "$OUTPUT_DIR/xeno_canto" \
    --delay 0.5

# Step 6: Search Macaulay Library
echo ""
echo "Step 6: Searching Macaulay Library..."
bioamla services ml search \
    --scientific-name "$BIRD_SPECIES" \
    --min-rating 4 \
    --max-results 5 \
    --format table

# Step 7: Download from Macaulay Library
echo ""
echo "Step 7: Downloading from Macaulay Library..."
bioamla services ml download \
    --scientific-name "$BIRD_SPECIES" \
    --min-rating 4 \
    --max-recordings 5 \
    --output-dir "$OUTPUT_DIR/macaulay"

# Step 8: Convert all downloads to consistent format
echo ""
echo "Step 8: Converting all audio to WAV format..."
bioamla audio convert "$OUTPUT_DIR/inat" wav --dataset
bioamla audio convert "$OUTPUT_DIR/xeno_canto" wav --dataset
bioamla audio convert "$OUTPUT_DIR/macaulay" wav --dataset

# Step 9: Merge datasets
echo ""
echo "Step 9: Merging datasets..."
bioamla dataset merge "$OUTPUT_DIR/merged_dataset" \
    "$OUTPUT_DIR/inat" "$OUTPUT_DIR/xeno_canto" "$OUTPUT_DIR/macaulay"

# Step 10: Generate license report
echo ""
echo "Step 10: Generating license report..."
bioamla dataset license "$OUTPUT_DIR/merged_dataset" \
    --output "$OUTPUT_DIR/LICENSE.txt"

# Step 11: Clear download caches (optional)
echo ""
echo "Step 11: Cache management..."
echo "To clear download caches, run:"
echo "  bioamla services clear-cache --all"
echo "  bioamla services clear-cache --xc    # Xeno-canto only"
echo "  bioamla services clear-cache --ml    # Macaulay only"

echo ""
echo "=== Data Acquisition Complete ==="
echo "Dataset saved to: $OUTPUT_DIR/merged_dataset"
echo ""
echo "Summary:"
echo "  - iNaturalist recordings: $OUTPUT_DIR/inat"
echo "  - Xeno-canto recordings: $OUTPUT_DIR/xeno_canto"
echo "  - Macaulay recordings: $OUTPUT_DIR/macaulay"
echo "  - Merged dataset: $OUTPUT_DIR/merged_dataset"
echo "  - License report: $OUTPUT_DIR/LICENSE.txt"
echo ""
echo "Note: Always check and comply with data source licenses"
echo "before using recordings for research or publications."
