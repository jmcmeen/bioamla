"""Clustering and dimensionality reduction commands."""

import click

from bioamla.exceptions import BioamlaError


@click.group()
def cluster() -> None:
    """Clustering and dimensionality reduction commands."""
    pass


def _load_npy(path: str, ndim: int | None = 2):
    """Load a .npy array, raising InvalidInputError if it can't be read or has the wrong shape."""
    import numpy as np

    from bioamla.exceptions import InvalidInputError

    try:
        array = np.load(path)
    except (OSError, ValueError, EOFError) as e:
        raise InvalidInputError(f"Cannot read {path} as a .npy array: {e}") from e
    if ndim is not None and array.ndim != ndim:
        raise InvalidInputError(f"{path} must be a {ndim}-D array, got shape {array.shape}")
    return array


def _save_npy(path: str, array) -> None:
    """Save a .npy array, creating parent directories; a failure is a ClickException."""
    from pathlib import Path

    import numpy as np

    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
    except OSError as e:
        raise click.ClickException(f"Failed to write {path}: {e}") from e


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
    from bioamla.cli.output import emit_file_result
    from bioamla.cluster import reduce_dimensions

    try:
        embeddings = _load_npy(embeddings_file)

        if not quiet:
            echo(f"Reducing {embeddings.shape[1]}D embeddings to {n_components}D using {method}...")

        reduced = reduce_dimensions(embeddings, method=method, n_components=n_components)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    _save_npy(output, reduced)
    if emit_file_result(embeddings_file, n_samples=len(embeddings), shape=list(reduced.shape)):
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
    import numpy as np

    from bioamla.cli.console import echo, print_header, print_success
    from bioamla.cli.output import emit_file_result
    from bioamla.cluster import AudioClusterer, ClusteringConfig

    try:
        embeddings = _load_npy(embeddings_file)

        if not quiet:
            echo(f"Clustering {len(embeddings)} samples using {method}...")

        config = ClusteringConfig(
            method=method,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            eps=eps,
        )
        clusterer = AudioClusterer(config=config)
        labels = clusterer.fit_predict(embeddings)
        labels = np.asarray(labels)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    _save_npy(output, labels)
    unique, counts = np.unique(labels, return_counts=True)
    if emit_file_result(
        embeddings_file,
        n_samples=len(labels),
        n_clusters=int(clusterer.n_clusters_),
        n_noise=int((labels == -1).sum()),
        cluster_sizes={str(int(u)): int(c) for u, c in zip(unique, counts, strict=True)},
    ):
        return
    if not quiet:
        print_header(f"Found {clusterer.n_clusters_} clusters")
        print_success(f"Saved cluster labels to: {output}")


@cluster.command("analyze")
@click.argument("embeddings_file")
@click.argument("labels_file")
@click.option("--output", "-o", help="Output JSON file for analysis results")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def cluster_analyze(embeddings_file: str, labels_file: str, output: str, quiet: bool) -> None:
    """Analyze cluster quality."""
    import json
    from pathlib import Path

    from bioamla.cli.console import print_header, print_kv, print_success
    from bioamla.cli.output import emit_file_result, to_jsonable
    from bioamla.cluster import analyze_clusters_summary

    try:
        embeddings = _load_npy(embeddings_file)
        labels = _load_npy(labels_file, ndim=1)
        analysis = analyze_clusters_summary(embeddings, labels)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    noise_pct = analysis.n_noise / analysis.n_samples * 100 if analysis.n_samples > 0 else 0
    analysis_dict = {**to_jsonable(analysis), "noise_percentage": noise_pct}

    if output:
        try:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(analysis_dict, indent=2), encoding="utf-8")
        except OSError as e:
            raise click.ClickException(f"Failed to write {output}: {e}") from e

    if emit_file_result(embeddings_file, labels_file=labels_file, **analysis_dict):
        return

    if not quiet:
        print_header("Cluster Analysis:")
        print_kv("  Clusters", f"{analysis.n_clusters}")
        print_kv("  Samples", f"{analysis.n_samples}")
        print_kv("  Noise", f"{analysis.n_noise} ({noise_pct:.1f}%)")
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
    import numpy as np

    from bioamla.cli.console import echo, print_header, print_success
    from bioamla.cli.output import emit_file_result
    from bioamla.cluster import detect_novelty

    try:
        embeddings = _load_npy(embeddings_file)
        known_labels = _load_npy(labels, ndim=1) if labels else None

        if not quiet:
            echo(f"Detecting novel sounds using {method}...")

        summary, is_novel, novelty_scores = detect_novelty(
            embeddings,
            known_labels=known_labels,
            method=method,
            threshold=threshold,
        )

        results = np.column_stack([is_novel.astype(int), novelty_scores])
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    _save_npy(output, results)
    if emit_file_result(
        embeddings_file,
        n_samples=len(is_novel),
        n_novel=int(summary.n_novel),
        novel_percentage=float(summary.novel_percentage),
        novel_indices=list(summary.novel_indices),
    ):
        return
    if not quiet:
        print_header(f"Found {summary.n_novel} novel samples ({summary.novel_percentage:.1f}%)")
        print_success(f"Saved novelty results to: {output}")
