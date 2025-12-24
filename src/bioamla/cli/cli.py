"""
BioAMLA CLI - Bioacoustic Machine Learning Analysis

A command-line interface for audio analysis, machine learning model training,
and bioacoustic research.
"""

import click

from bioamla import __version__

from .commands import (
    annotation,
    audio,
    cluster,
    config,
    dataset,
    detect,
    examples,
    indices,
    learn,
    models,
    pipeline,
    realtime,
    services,
)


@click.group()
@click.version_option(version=__version__, prog_name="bioamla")
def cli():
    """BioAMLA - Bioacoustic Machine Learning Analysis

    A comprehensive toolkit for bioacoustic research and analysis.

    \b
    Quick start:
        bioamla audio info recording.wav     # Get audio file info
        bioamla models predict ast audio.wav # Run ML inference
        bioamla indices compute recording.wav # Compute acoustic indices

    \b
    Use 'bioamla COMMAND --help' for more information on a command.
    """
    pass


# Register command groups
cli.add_command(annotation)
cli.add_command(audio)
cli.add_command(cluster)
cli.add_command(config)
cli.add_command(dataset)
cli.add_command(detect)
cli.add_command(examples)
cli.add_command(indices)
cli.add_command(learn)
cli.add_command(models)
cli.add_command(pipeline)
cli.add_command(realtime)
cli.add_command(services)


# Top-level convenience commands
@cli.command("devices")
def devices():
    """List available compute devices (GPU/CPU)."""
    import torch

    click.echo("Compute Devices:")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            click.echo(f"  [cuda:{i}] {props.name} ({memory_gb:.1f} GB)")
    else:
        click.echo("  No CUDA devices available")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        click.echo("  [mps] Apple Metal Performance Shaders")

    click.echo("  [cpu] CPU")


@cli.command("version")
def version():
    """Show version information."""
    import platform
    import sys

    click.echo(f"bioamla {__version__}")
    click.echo(f"Python {sys.version}")
    click.echo(f"Platform: {platform.platform()}")

    try:
        import torch

        click.echo(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            click.echo(f"CUDA: {torch.version.cuda}")
    except ImportError:
        pass


if __name__ == "__main__":
    cli()
