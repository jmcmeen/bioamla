#!/usr/bin/env bash
#
# Workflow 5: embeddings → clustering → novelty
# ---------------------------------------------
# Unsupervised exploration of a sound collection: extract AST embeddings for a
# folder of clips, reduce dimensionality, cluster, and flag novel/outlier sounds
# worth a closer look. Useful before you have labels.
#
# Needs: a directory of audio clips (set CLIPS) and a model (set MODEL). No
# labels required.

set -euo pipefail

OUT=./out/04_embedding_clustering
mkdir -p "$OUT"

CLIPS=./out/04_embedding_clustering/clips   # <-- folder of .wav clips
MODEL="MIT/ast-finetuned-audioset-10-10-0.4593"

if [[ ! -d "$CLIPS" ]]; then
  echo "Set CLIPS to a directory of audio clips (e.g. from 'audio segment')." >&2
  exit 1
fi

# 1. Extract embeddings for every clip (one .npy of shape [n_clips, dim]).
bioamla batch models embed --input-dir "$CLIPS" --output-dir "$OUT/embeddings" --model "$MODEL"
EMB="$OUT/embeddings/embeddings.npy"

# 2. Reduce to 2-D for plotting/inspection (UMAP preserves local structure).
bioamla cluster reduce "$EMB" -o "$OUT/reduced.npy" --method umap --n-components 2

# 3. Cluster the full-dimensional embeddings. HDBSCAN (the default) finds the
#    cluster count on its own and marks outliers as noise — ideal when you don't
#    know how many classes there are. Use --method kmeans --n-clusters N if you do.
bioamla cluster cluster "$EMB" -o "$OUT/clusters.npy" --method hdbscan --min-cluster-size 5

# 4. Flag novel/outlier sounds — candidates for new classes or rare events.
bioamla cluster novelty "$EMB" -o "$OUT/novelty.npy" --method isolation_forest --threshold 0.9

echo "Done. Clusters: $OUT/clusters.npy, novelty: $OUT/novelty.npy"
echo "Tip: clips flagged novel are good candidates to annotate and add as a new class."
