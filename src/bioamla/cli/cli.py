"""
BioAMLA CLI - Bioacoustic Machine Learning Analysis
"""

import click

from bioamla import __version__

from .commands import (
    annotation,
    audio,
    batch,
    catalogs,
    cluster,
    config,
    dataset,
    detect,
    indices,
    models,
)


@click.group()
@click.version_option(version=__version__, prog_name="bioamla")
def cli() -> None:
    """BioAMLA - Bioacoustic & Machine Learning Applications

    Use 'bioamla COMMAND --help' for more information on a command.
    """
    pass


# Register command groups
cli.add_command(annotation)
cli.add_command(audio)
cli.add_command(batch)
cli.add_command(cluster)
cli.add_command(config)
cli.add_command(dataset)
cli.add_command(detect)
cli.add_command(indices)
cli.add_command(models)
cli.add_command(catalogs)


# Top-level convenience commands
@cli.command("devices")
def devices() -> None:
    """List available compute devices (GPU/CPU)."""
    from bioamla.services.util import UtilityService

    result = UtilityService().get_device_info()

    if not result.success:
        click.echo(f"Error: {result.error}", err=True)
        raise SystemExit(1)

    data = result.data
    click.echo("Compute Devices:")

    if not data.cuda_available:
        click.echo("  No CUDA devices available")

    for device in data.devices:
        if device.memory_gb:
            click.echo(f"  [{device.device_id}] {device.name} ({device.memory_gb} GB)")
        else:
            click.echo(f"  [{device.device_id}] {device.name}")


if __name__ == "__main__":
    cli()
