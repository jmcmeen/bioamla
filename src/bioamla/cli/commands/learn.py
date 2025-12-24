"""Active learning commands for efficient annotation."""

from typing import Optional

import click

from bioamla.core.files import TextFile


@click.group()
def learn():
    """Active learning commands for efficient annotation."""
    pass


@learn.command("init")
@click.argument("predictions_csv", type=click.Path(exists=True))
@click.argument("output_state", type=click.Path())
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["entropy", "least_confidence", "margin", "random", "hybrid"]),
    default="entropy",
    help="Sampling strategy",
)
@click.option(
    "--labeled-csv",
    type=click.Path(exists=True),
    help="CSV file with pre-labeled samples (id,label columns)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def learn_init(
    predictions_csv: str, output_state: str, strategy: str, labeled_csv: Optional[str], quiet: bool
):
    """Initialize active learning session from predictions."""
    import csv

    from bioamla.core.active_learning import (
        ActiveLearner,
        HybridSampler,
        RandomSampler,
        UncertaintySampler,
        create_samples_from_predictions,
    )

    if strategy == "random":
        sampler = RandomSampler()
    elif strategy == "hybrid":
        sampler = HybridSampler()
    else:
        sampler = UncertaintySampler(strategy=strategy)

    learner = ActiveLearner(sampler=sampler)

    samples = create_samples_from_predictions(predictions_csv)
    learner.add_unlabeled(samples)

    if not quiet:
        click.echo(f"Loaded {len(samples)} samples from {predictions_csv}")

    if labeled_csv:
        labeled_samples = []
        with TextFile(labeled_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f.handle)
            for row in reader:
                sample_id = row.get("id") or row.get("sample_id")
                label = row.get("label")
                if sample_id and label:
                    if sample_id in learner.unlabeled_pool:
                        sample = learner.unlabeled_pool[sample_id]
                        sample.label = label
                        labeled_samples.append(sample)

        if labeled_samples:
            learner.add_labeled(labeled_samples)
            if not quiet:
                click.echo(f"Added {len(labeled_samples)} pre-labeled samples")

    learner.save_state(output_state)

    if not quiet:
        click.echo("\nActive learning session initialized:")
        click.echo(f"  Strategy: {strategy}")
        click.echo(f"  Unlabeled samples: {learner.state.total_unlabeled}")
        click.echo(f"  Labeled samples: {learner.state.total_labeled}")
        click.echo(f"  State saved to: {output_state}")


@learn.command("query")
@click.argument("state_file", type=click.Path(exists=True))
@click.option("--n-samples", "-n", default=10, help="Number of samples to query")
@click.option("--output", "-o", type=click.Path(), help="Output CSV for query results")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def learn_query(state_file: str, n_samples: int, output: Optional[str], quiet: bool):
    """Query samples for annotation from active learning session."""
    import csv
    import json
    from pathlib import Path

    from bioamla.core.active_learning import (
        ActiveLearner,
        UncertaintySampler,
    )

    with TextFile(state_file, mode="r", encoding="utf-8") as f:
        json.load(f.handle)

    sampler = UncertaintySampler(strategy="entropy")

    learner = ActiveLearner.load_state(state_file, sampler=sampler)

    if learner.state.total_unlabeled == 0:
        click.echo("No unlabeled samples remaining!")
        return

    queried = learner.query(n_samples=n_samples, update_predictions=False)

    if not quiet:
        click.echo(f"\nQueried {len(queried)} samples (iteration {learner.state.iteration}):")
        for i, sample in enumerate(queried, 1):
            conf_str = f"{sample.confidence:.3f}" if sample.confidence else "N/A"
            click.echo(f"  {i}. {sample.id}")
            click.echo(f"     File: {sample.filepath}")
            click.echo(f"     Time: {sample.start_time:.2f}s - {sample.end_time:.2f}s")
            click.echo(f"     Predicted: {sample.predicted_label or 'N/A'} (conf: {conf_str})")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with TextFile(output, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "id",
                "filepath",
                "start_time",
                "end_time",
                "predicted_label",
                "confidence",
                "label",
            ]
            writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
            writer.writeheader()
            for sample in queried:
                writer.writerow(
                    {
                        "id": sample.id,
                        "filepath": sample.filepath,
                        "start_time": sample.start_time,
                        "end_time": sample.end_time,
                        "predicted_label": sample.predicted_label or "",
                        "confidence": sample.confidence or "",
                        "label": "",
                    }
                )

        if not quiet:
            click.echo(f"\nQuery results saved to: {output}")
            click.echo("Fill in the 'label' column and use 'learn annotate' to import.")

    learner.save_state(state_file)


@learn.command("annotate")
@click.argument("state_file", type=click.Path(exists=True))
@click.argument("annotations_csv", type=click.Path(exists=True))
@click.option("--annotator", "-a", default="unknown", help="Annotator identifier")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def learn_annotate(state_file: str, annotations_csv: str, annotator: str, quiet: bool):
    """Import annotations into active learning session."""
    import csv

    from bioamla.core.active_learning import ActiveLearner, UncertaintySampler

    sampler = UncertaintySampler(strategy="entropy")
    learner = ActiveLearner.load_state(state_file, sampler=sampler)

    annotations_imported = 0
    with TextFile(annotations_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f.handle)
        for row in reader:
            sample_id = row.get("id") or row.get("sample_id")
            label = row.get("label", "").strip()

            if not sample_id or not label:
                continue

            if sample_id in learner.unlabeled_pool:
                sample = learner.unlabeled_pool[sample_id]
                learner.teach(sample, label, annotator=annotator)
                annotations_imported += 1

    learner.save_state(state_file)

    if not quiet:
        click.echo(f"\nImported {annotations_imported} annotations")
        click.echo(f"  Annotator: {annotator}")
        click.echo(f"  Total labeled: {learner.state.total_labeled}")
        click.echo(f"  Remaining unlabeled: {learner.state.total_unlabeled}")
        click.echo(f"  Labels per class: {learner.state.labels_per_class}")


@learn.command("status")
@click.argument("state_file", type=click.Path(exists=True))
def learn_status(state_file: str):
    """Show status of active learning session."""
    from bioamla.core.active_learning import (
        ActiveLearner,
        UncertaintySampler,
        summarize_annotation_session,
    )

    sampler = UncertaintySampler(strategy="entropy")
    learner = ActiveLearner.load_state(state_file, sampler=sampler)

    summary = summarize_annotation_session(learner)

    click.echo("\nActive Learning Session Status")
    click.echo("=" * 40)
    click.echo(f"Iteration: {summary['iteration']}")
    click.echo(f"Total labeled: {summary['total_labeled']}")
    click.echo(f"Total unlabeled: {summary['total_unlabeled']}")
    click.echo(f"Total annotations: {summary['total_annotations']}")

    if summary["labels_per_class"]:
        click.echo("\nLabels per class:")
        for label, count in sorted(summary["labels_per_class"].items()):
            click.echo(f"  {label}: {count}")

    if summary["total_annotation_time_seconds"] > 0:
        click.echo("\nAnnotation statistics:")
        click.echo(f"  Total time: {summary['total_annotation_time_seconds']:.1f}s")
        click.echo(f"  Rate: {summary['annotations_per_hour']:.1f} annotations/hour")

    if summary["class_balance_ratio"] > 0:
        click.echo(f"  Class balance ratio: {summary['class_balance_ratio']:.2f}")


@learn.command("export")
@click.argument("state_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["csv", "raven"]),
    default="csv",
    help="Output format",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def learn_export(state_file: str, output_file: str, fmt: str, quiet: bool):
    """Export labeled samples from active learning session."""
    from bioamla.core.active_learning import ActiveLearner, UncertaintySampler, export_annotations

    sampler = UncertaintySampler(strategy="entropy")
    learner = ActiveLearner.load_state(state_file, sampler=sampler)

    export_annotations(learner, output_file, format=fmt)

    if not quiet:
        click.echo(f"\nExported {learner.state.total_labeled} annotations to {output_file}")
        click.echo(f"  Format: {fmt}")


@learn.command("simulate")
@click.argument("predictions_csv", type=click.Path(exists=True))
@click.argument("ground_truth_csv", type=click.Path(exists=True))
@click.option("--n-iterations", "-n", default=10, help="Number of iterations")
@click.option("--batch-size", "-b", default=10, help="Samples per iteration")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["entropy", "least_confidence", "margin", "random", "hybrid"]),
    default="entropy",
    help="Sampling strategy",
)
@click.option("--output", "-o", type=click.Path(), help="Output CSV for simulation results")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def learn_simulate(
    predictions_csv: str,
    ground_truth_csv: str,
    n_iterations: int,
    batch_size: int,
    strategy: str,
    output: Optional[str],
    quiet: bool,
):
    """Simulate active learning loop using ground truth labels."""
    import csv
    from pathlib import Path

    from bioamla.core.active_learning import (
        ActiveLearner,
        HybridSampler,
        RandomSampler,
        SimulatedOracle,
        UncertaintySampler,
        create_samples_from_predictions,
    )

    ground_truth = {}
    with TextFile(ground_truth_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f.handle)
        for row in reader:
            sample_id = row.get("id") or row.get("sample_id")
            label = row.get("label", "").strip()
            if sample_id and label:
                ground_truth[sample_id] = label

    if not ground_truth:
        click.echo("Error: No ground truth labels found in CSV")
        return

    if strategy == "random":
        sampler = RandomSampler()
    elif strategy == "hybrid":
        sampler = HybridSampler()
    else:
        sampler = UncertaintySampler(strategy=strategy)

    oracle = SimulatedOracle(ground_truth=ground_truth)

    learner = ActiveLearner(sampler=sampler)
    samples = create_samples_from_predictions(predictions_csv)

    samples = [s for s in samples if s.id in ground_truth]
    learner.add_unlabeled(samples)

    if not quiet:
        click.echo("\nSimulating active learning:")
        click.echo(f"  Strategy: {strategy}")
        click.echo(f"  Samples: {len(samples)}")
        click.echo(f"  Iterations: {n_iterations}")
        click.echo(f"  Batch size: {batch_size}")
        click.echo()

    results = []
    for iteration in range(n_iterations):
        if learner.state.total_unlabeled == 0:
            break

        queried = learner.query(n_samples=batch_size, update_predictions=False)

        for sample in queried:
            try:
                label = oracle.annotate(sample)
                learner.teach(sample, label, annotator="oracle")
            except ValueError:
                pass

        result = {
            "iteration": iteration + 1,
            "total_labeled": learner.state.total_labeled,
            "total_unlabeled": learner.state.total_unlabeled,
        }
        results.append(result)

        if not quiet:
            click.echo(f"  Iteration {iteration + 1}: {learner.state.total_labeled} labeled")

    if not quiet:
        click.echo("\nSimulation complete:")
        click.echo(f"  Final labeled: {learner.state.total_labeled}")
        click.echo(f"  Labels per class: {learner.state.labels_per_class}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with TextFile(output, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = ["iteration", "total_labeled", "total_unlabeled"]
            writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        if not quiet:
            click.echo(f"  Results saved to: {output}")
