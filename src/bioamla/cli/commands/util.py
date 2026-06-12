"""General-purpose file utilities (download / unzip / zip).

These are not dataset-specific — they wrap the simple file helpers in
:mod:`bioamla.common.files` — so they live in their own ``util`` group rather
than under ``dataset``.
"""

import click

from bioamla.exceptions import BioamlaError


@click.group()
def util() -> None:
    """General file utilities (download, unzip, zip)."""
    pass


@util.command("download")
@click.argument("url", required=True)
@click.argument("output_dir", required=False, default=".")
def util_download(url: str, output_dir: str) -> None:
    """Download a file from the specified URL to the target directory."""
    import os
    from urllib.parse import urlparse

    from bioamla.common.files import download_file

    if output_dir == ".":
        output_dir = os.getcwd()

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path) or "downloaded_file"
    output_path = os.path.join(output_dir, filename)

    try:
        download_file(url, output_path)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    except OSError as e:
        raise click.ClickException(f"Download failed: {e}") from e

    click.echo(f"Downloaded to {output_path}")


@util.command("unzip")
@click.argument("file_path")
@click.argument("output_path", required=False, default=".")
def util_unzip(file_path: str, output_path: str) -> None:
    """Extract a ZIP archive to the specified output directory."""
    import os

    from bioamla.common.files import extract_zip_file

    if output_path == ".":
        output_path = os.getcwd()

    try:
        extract_zip_file(file_path, output_path)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    except OSError as e:
        raise click.ClickException(f"Extraction failed: {e}") from e

    click.echo(f"Extracted to {output_path}")


@util.command("zip")
@click.argument("source_path")
@click.argument("output_file")
def util_zip(source_path: str, output_file: str) -> None:
    """Create a ZIP archive from a file or directory."""
    from pathlib import Path

    from bioamla.common.files import create_zip_file, zip_directory

    try:
        if Path(source_path).is_dir():
            zip_directory(source_path, output_file)
        else:
            create_zip_file([source_path], output_file)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    except OSError as e:
        raise click.ClickException(f"ZIP creation failed: {e}") from e

    click.echo(f"Created {output_file}")
