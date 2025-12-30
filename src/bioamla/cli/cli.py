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


if __name__ == "__main__":
    cli()
