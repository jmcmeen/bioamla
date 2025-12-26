"""Clustering and dimensionality reduction commands."""

import click


@click.group()
def cluster() -> None:
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
def cluster_reduce(embeddings_file: str, output: str, method: str, n_components: int, quiet: bool) -> None:
    """Reduce dimensionality of embeddings."""
    import numpy as np

    from bioamla.cli.service_helpers import handle_result, services

    embeddings = np.load(embeddings_file)

    if not quiet:
        click.echo(
            f"Reducing {embeddings.shape[1]}D embeddings to {n_components}D using {method}..."
        )

    result = services.clustering.reduce_dimensions(
        embeddings, method=method, n_components=n_components, output_path=output
    )
    handle_result(result)

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
) -> None:
    """Cluster embeddings."""
    import numpy as np

    from bioamla.cli.service_helpers import handle_result, services

    embeddings = np.load(embeddings_file)

    if not quiet:
        click.echo(f"Clustering {len(embeddings)} samples using {method}...")

    result = services.clustering.cluster(
        embeddings,
        method=method,
        n_clusters=n_clusters,
        min_samples=min_samples,
        eps=eps,
    )
    cluster_result = handle_result(result)

    labels = np.array(cluster_result.labels)
    services.file.write_npy(output, labels)

    if not quiet:
        click.echo(f"Found {cluster_result.n_clusters} clusters")
        click.echo(f"Saved cluster labels to: {output}")


@cluster.command("analyze")
@click.argument("embeddings_file")
@click.argument("labels_file")
@click.option("--output", "-o", help="Output JSON file for analysis results")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def cluster_analyze(embeddings_file: str, labels_file: str, output: str, quiet: bool) -> None:
    """Analyze cluster quality."""
    from typing import Any

    import numpy as np

    from bioamla.cli.service_helpers import handle_result, services

    embeddings = np.load(embeddings_file)
    labels = np.load(labels_file)

    result = services.clustering.analyze_clusters(embeddings, labels)
    analysis = handle_result(result)

    if not quiet:
        click.echo("Cluster Analysis:")
        click.echo(f"  Clusters: {analysis.n_clusters}")
        click.echo(f"  Samples: {analysis.n_samples}")
        noise_pct = analysis.n_noise / analysis.n_samples * 100 if analysis.n_samples > 0 else 0
        click.echo(f"  Noise: {analysis.n_noise} ({noise_pct:.1f}%)")
        click.echo(f"  Silhouette Score: {analysis.silhouette_score:.4f}")
        click.echo(f"  Calinski-Harabasz Score: {analysis.calinski_harabasz_score:.2f}")

    if output:
        def convert_numpy(obj: Any) -> Any:
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

        analysis_dict = {
            "n_clusters": analysis.n_clusters,
            "n_samples": analysis.n_samples,
            "n_noise": analysis.n_noise,
            "noise_percentage": noise_pct,
            "silhouette_score": analysis.silhouette_score,
            "calinski_harabasz_score": analysis.calinski_harabasz_score,
            "cluster_stats": convert_numpy(analysis.cluster_stats),
        }
        services.file.write_json(output, analysis_dict)
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
) -> None:
    """Detect novel sounds in embeddings."""
    import numpy as np

    from bioamla.cli.service_helpers import handle_result, services

    embeddings = np.load(embeddings_file)
    known_labels = np.load(labels) if labels else None

    if not quiet:
        click.echo(f"Detecting novel sounds using {method}...")

    result = services.clustering.detect_novelty(
        embeddings,
        known_labels=known_labels,
        method=method,
        threshold=threshold,
    )
    novelty_result = handle_result(result)

    is_novel = novelty_result.metadata.get("is_novel", np.array([]))
    novelty_scores = novelty_result.metadata.get("novelty_scores", np.array([]))

    results = np.column_stack([is_novel.astype(int), novelty_scores])
    services.file.write_npy(output, results)

    n_novel = novelty_result.data.n_novel
    if not quiet:
        click.echo(f"Found {n_novel} novel samples ({novelty_result.data.novel_percentage:.1f}%)")
        click.echo(f"Saved novelty results to: {output}")
