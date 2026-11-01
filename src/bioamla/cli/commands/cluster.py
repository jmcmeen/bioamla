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
def cluster_reduce(
    embeddings_file: str, output: str, method: str, n_components: int, quiet: bool
) -> None:
    """Reduce dimensionality of embeddings."""
    from bioamla.cli.console import echo, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    if not quiet:
        echo(f"Reducing embeddings to {n_components}D using {method}...")
    result = run_operation(
        "cluster.reduce",
        embeddings_file=embeddings_file,
        output=output,
        method=method,
        n_components=n_components,
    )
    if emit_file_result(embeddings_file, n_samples=result.n_samples, shape=result.shape):
        return
    if not quiet:
        print_success(f"Saved reduced embeddings to: {output}")


@cluster.command("fit")
@click.argument("embeddings_file")
@click.option("--output", "-o", required=True, help="Output file for cluster labels")
@click.option(
    "--method",
    "-m",
    type=click.Choice(["hdbscan", "kmeans", "dbscan", "agglomerative"]),
    default="hdbscan",
    help="Clustering method (hdbscan finds clusters without a preset count)",
)
@click.option(
    "--n-clusters",
    "-k",
    type=int,
    default=10,
    help="Number of clusters (for k-means/agglomerative)",
)
@click.option("--min-cluster-size", type=int, default=5, help="Minimum cluster size (HDBSCAN)")
@click.option("--eps", type=float, default=0.5, help="DBSCAN epsilon")
@click.option("--min-samples", type=int, default=5, help="Minimum samples per cluster")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def cluster_fit(
    embeddings_file: str,
    output: str,
    method: str,
    n_clusters: int,
    min_cluster_size: int,
    eps: float,
    min_samples: int,
    quiet: bool,
) -> None:
    """Cluster embeddings."""
    from bioamla.cli.console import echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    if not quiet:
        echo(f"Clustering embeddings using {method}...")
    result = run_operation(
        "cluster.fit",
        embeddings_file=embeddings_file,
        output=output,
        method=method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        eps=eps,
        min_samples=min_samples,
    )
    if emit_file_result(
        embeddings_file,
        n_samples=result.n_samples,
        n_clusters=result.n_clusters,
        n_noise=result.n_noise,
        cluster_sizes=result.cluster_sizes,
    ):
        return
    if not quiet:
        print_header(f"Found {result.n_clusters} clusters")
        print_success(f"Saved cluster labels to: {output}")


@cluster.command("analyze")
@click.argument("embeddings_file")
@click.argument("labels_file")
@click.option("--output", "-o", help="Output JSON file for analysis results")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def cluster_analyze(embeddings_file: str, labels_file: str, output: str, quiet: bool) -> None:
    """Analyze cluster quality."""
    from bioamla.cli.console import print_header, print_kv, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    report = run_operation(
        "cluster.analyze", embeddings_file=embeddings_file, labels_file=labels_file, output=output
    )
    analysis = report.analysis
    data = {k: v for k, v in report.to_dict().items() if k not in ("file", "labels_file", "output")}
    if emit_file_result(embeddings_file, labels_file=labels_file, **data):
        return

    if not quiet:
        print_header("Cluster Analysis:")
        print_kv("  Clusters", f"{analysis.n_clusters}")
        print_kv("  Samples", f"{analysis.n_samples}")
        print_kv("  Noise", f"{analysis.n_noise} ({report.noise_percentage:.1f}%)")
        print_kv("  Silhouette Score", f"{analysis.silhouette_score:.4f}")
        print_kv("  Calinski-Harabasz Score", f"{analysis.calinski_harabasz_score:.2f}")
        if output:
            print_success(f"Saved analysis to: {output}")


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
    from bioamla.cli.console import echo, print_header, print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_file_result

    if not quiet:
        echo(f"Detecting novel sounds using {method}...")
    report = run_operation(
        "cluster.novelty",
        embeddings_file=embeddings_file,
        output=output,
        method=method,
        threshold=threshold,
        labels_file=labels,
    )
    if emit_file_result(
        embeddings_file,
        n_samples=report.n_samples,
        n_novel=report.n_novel,
        novel_percentage=report.novel_percentage,
        novel_indices=report.novel_indices,
    ):
        return
    if not quiet:
        print_header(f"Found {report.n_novel} novel samples ({report.novel_percentage:.1f}%)")
        print_success(f"Saved novelty results to: {output}")
