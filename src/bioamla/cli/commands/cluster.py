"""Clustering and dimensionality reduction commands."""

import click

from bioamla.core.files import TextFile


@click.group()
def cluster():
    """Clustering and dimensionality reduction commands."""
    pass


@cluster.command("reduce")
@click.argument("embeddings_file")
@click.option("--output", "-o", required=True, help="Output file for reduced embeddings")
@click.option(
    "--method",
    "-m",
    type=click.Choice(["umap", "tsne", "pca"]),
    default="pca",
    help="Reduction method",
)
@click.option("--n-components", "-n", type=int, default=2, help="Number of output dimensions")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def cluster_reduce(embeddings_file: str, output: str, method: str, n_components: int, quiet: bool):
    """Reduce dimensionality of embeddings."""
    import numpy as np

    from bioamla.core.analysis.clustering import reduce_dimensions

    embeddings = np.load(embeddings_file)

    if not quiet:
        click.echo(
            f"Reducing {embeddings.shape[1]}D embeddings to {n_components}D using {method}..."
        )

    reduced = reduce_dimensions(embeddings, method=method, n_components=n_components)

    np.save(output, reduced)

    if not quiet:
        click.echo(f"Saved reduced embeddings to: {output}")


@cluster.command("cluster")
@click.argument("embeddings_file")
@click.option("--output", "-o", required=True, help="Output file for cluster labels")
@click.option(
    "--method",
    "-m",
    type=click.Choice(["kmeans", "dbscan", "agglomerative"]),
    default="kmeans",
    help="Clustering method",
)
@click.option(
    "--n-clusters",
    "-k",
    type=int,
    default=10,
    help="Number of clusters (for k-means/agglomerative)",
)
@click.option("--eps", type=float, default=0.5, help="DBSCAN epsilon")
@click.option("--min-samples", type=int, default=5, help="Minimum samples per cluster")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def cluster_cluster(
    embeddings_file: str,
    output: str,
    method: str,
    n_clusters: int,
    eps: float,
    min_samples: int,
    quiet: bool,
):
    """Cluster embeddings."""
    import numpy as np

    from bioamla.core.analysis.clustering import AudioClusterer, ClusteringConfig

    embeddings = np.load(embeddings_file)

    config = ClusteringConfig(
        method=method,
        n_clusters=n_clusters,
        eps=eps,
        min_samples=min_samples,
    )
    clusterer = AudioClusterer(config=config)

    if not quiet:
        click.echo(f"Clustering {len(embeddings)} samples using {method}...")

    labels = clusterer.fit_predict(embeddings)

    np.save(output, labels)

    if not quiet:
        click.echo(f"Found {clusterer.n_clusters_} clusters")
        click.echo(f"Saved cluster labels to: {output}")


@cluster.command("analyze")
@click.argument("embeddings_file")
@click.argument("labels_file")
@click.option("--output", "-o", help="Output JSON file for analysis results")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def cluster_analyze(embeddings_file: str, labels_file: str, output: str, quiet: bool):
    """Analyze cluster quality."""
    import json
    from pathlib import Path

    import numpy as np

    from bioamla.core.analysis.clustering import analyze_clusters

    embeddings = np.load(embeddings_file)
    labels = np.load(labels_file)

    analysis = analyze_clusters(embeddings, labels)

    if not quiet:
        click.echo("Cluster Analysis:")
        click.echo(f"  Clusters: {analysis['n_clusters']}")
        click.echo(f"  Samples: {analysis['n_samples']}")
        click.echo(f"  Noise: {analysis['n_noise']} ({analysis['noise_percentage']:.1f}%)")
        click.echo(f"  Silhouette Score: {analysis['silhouette_score']:.4f}")
        click.echo(f"  Calinski-Harabasz Score: {analysis['calinski_harabasz_score']:.2f}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        with TextFile(output, mode="w", encoding="utf-8") as f:
            json.dump(convert_numpy(analysis), f.handle, indent=2)
        if not quiet:
            click.echo(f"Saved analysis to: {output}")


@cluster.command("novelty")
@click.argument("embeddings_file")
@click.option("--output", "-o", required=True, help="Output file for novelty results")
@click.option(
    "--method",
    "-m",
    type=click.Choice(["distance", "isolation_forest", "lof"]),
    default="distance",
    help="Novelty detection method",
)
@click.option("--threshold", type=float, help="Novelty threshold")
@click.option("--labels", help="Optional cluster labels file")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def cluster_novelty(
    embeddings_file: str, output: str, method: str, threshold: float, labels: str, quiet: bool
):
    """Detect novel sounds in embeddings."""
    import numpy as np

    from bioamla.core.analysis.clustering import discover_novel_sounds

    embeddings = np.load(embeddings_file)
    known_labels = np.load(labels) if labels else None

    if not quiet:
        click.echo(f"Detecting novel sounds using {method}...")

    is_novel, scores = discover_novel_sounds(
        embeddings,
        known_labels=known_labels,
        method=method,
        threshold=threshold,
        return_scores=True,
    )

    results = np.column_stack([is_novel.astype(int), scores])
    np.save(output, results)

    n_novel = is_novel.sum()
    if not quiet:
        click.echo(f"Found {n_novel} novel samples ({100 * n_novel / len(embeddings):.1f}%)")
        click.echo(f"Saved novelty results to: {output}")
