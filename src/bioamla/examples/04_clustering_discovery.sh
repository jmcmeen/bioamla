#!/bin/bash
# =============================================================================
# Clustering and Sound Discovery Workflow
# =============================================================================
# PURPOSE: Discover and group unknown sound types in recordings using
#          unsupervised learning techniques. Useful for exploring new datasets
#          or finding rare/unknown species vocalizations.
#
# FEATURES DEMONSTRATED:
#   - Audio embedding extraction using pretrained models
#   - Dimensionality reduction (UMAP, t-SNE, PCA)
#   - Clustering (HDBSCAN, K-means, DBSCAN)
#   - Novelty detection for finding unusual sounds
#   - Cluster analysis and visualization
#
# INPUT: Directory of audio recordings (unlabeled or partially labeled)
# OUTPUT: Cluster assignments, reduced embeddings, and novelty scores
# =============================================================================

set -e  # Exit on error

# Configuration
AUDIO_DIR="${1:-./raw_recordings}"
OUTPUT_DIR="./discovery_results"
MODEL_PATH="MIT/ast-finetuned-audioset-10-10-0.4593"

echo "=== Clustering and Sound Discovery Workflow ==="
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Extract embeddings from audio files
# Embeddings capture acoustic features in a high-dimensional space
echo "Step 1: Extracting audio embeddings..."
bioamla models embed "$AUDIO_DIR" \
    --model-type ast \
    --model-path "$MODEL_PATH" \
    --output "$OUTPUT_DIR/embeddings.npy" \
    --batch \
    --layer -1

# Step 2: Reduce dimensionality for visualization and clustering

# UMAP - good for preserving local and global structure
echo ""
echo "Step 2a: Reducing dimensions with UMAP..."
bioamla cluster reduce "$OUTPUT_DIR/embeddings.npy" \
    --output "$OUTPUT_DIR/umap_embeddings.npy" \
    --method umap \
    --n-components 2

# t-SNE - good for visualization
echo ""
echo "Step 2b: Reducing dimensions with t-SNE..."
bioamla cluster reduce "$OUTPUT_DIR/embeddings.npy" \
    --output "$OUTPUT_DIR/tsne_embeddings.npy" \
    --method tsne \
    --n-components 2

# PCA - fast, linear reduction
echo ""
echo "Step 2c: Reducing dimensions with PCA..."
bioamla cluster reduce "$OUTPUT_DIR/embeddings.npy" \
    --output "$OUTPUT_DIR/pca_embeddings.npy" \
    --method pca \
    --n-components 10

# Step 3: Cluster the embeddings

# DBSCAN - density-based, finds clusters of varying shapes
echo ""
echo "Step 3a: Clustering with DBSCAN..."
bioamla cluster cluster "$OUTPUT_DIR/umap_embeddings.npy" \
    --output "$OUTPUT_DIR/dbscan_labels.npy" \
    --method dbscan \
    --eps 0.5 \
    --min-samples 3

# K-means - requires specifying number of clusters
echo ""
echo "Step 3b: Clustering with K-means (k=10)..."
bioamla cluster cluster "$OUTPUT_DIR/pca_embeddings.npy" \
    --output "$OUTPUT_DIR/kmeans_labels.npy" \
    --method kmeans \
    --n-clusters 10

# Agglomerative - hierarchical clustering
echo ""
echo "Step 3c: Clustering with Agglomerative..."
bioamla cluster cluster "$OUTPUT_DIR/umap_embeddings.npy" \
    --output "$OUTPUT_DIR/agglomerative_labels.npy" \
    --method agglomerative \
    --n-clusters 10

# Step 4: Analyze clusters
echo ""
echo "Step 4: Analyzing cluster structure..."
bioamla cluster analyze "$OUTPUT_DIR/embeddings.npy" "$OUTPUT_DIR/dbscan_labels.npy" \
    --output "$OUTPUT_DIR/cluster_analysis.json"

# Step 5: Novelty detection
# Find unusual or rare sounds that don't fit known patterns
echo ""
echo "Step 5: Detecting novel/unusual sounds..."
bioamla cluster novelty "$OUTPUT_DIR/embeddings.npy" \
    --output "$OUTPUT_DIR/novelty_scores.npy" \
    --method isolation_forest \
    --threshold 0.8

echo ""
echo "=== Clustering and Discovery Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Output files:"
echo "  - embeddings.npy: Raw audio embeddings"
echo "  - umap_embeddings.npy: 2D UMAP projection for visualization"
echo "  - tsne_embeddings.npy: 2D t-SNE projection for visualization"
echo "  - pca_embeddings.npy: 10D PCA projection for downstream tasks"
echo "  - dbscan_labels.npy: DBSCAN cluster assignments"
echo "  - kmeans_labels.npy: K-means cluster assignments"
echo "  - agglomerative_labels.npy: Agglomerative cluster assignments"
echo "  - cluster_analysis.json: Cluster statistics and metrics"
echo "  - novelty_scores.npy: Anomaly scores (higher = more unusual)"
echo ""
echo "Next steps:"
echo "  1. Visualize 2D embeddings colored by cluster labels"
echo "  2. Review high novelty score samples manually"
echo "  3. Listen to samples from each cluster to identify patterns"
