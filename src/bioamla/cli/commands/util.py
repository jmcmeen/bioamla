"""General-purpose file utilities (download / unzip / zip).

These are not dataset-specific — they wrap the simple file helpers in
:mod:`bioamla.common.files` — so they live in their own ``util`` group rather
than under ``dataset``.
"""

import os

import click


@click.group()
def util() -> None:
    """General file utilities (download, unzip, zip)."""
    pass


@util.command("download")
@click.argument("url", required=True)
@click.argument("output_dir", required=False, default=".")
def util_download(url: str, output_dir: str) -> None:
    """Download a file from the specified URL to the target directory."""
    from bioamla.cli.console import print_success
    from bioamla.cli.logging_setup import configure_cli_logging
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    configure_cli_logging()
    folder = os.getcwd() if output_dir == "." else output_dir
    result = run_operation("util.download", url=url, output_dir=folder)
    if emit_result(result):
        return
    print_success(f"Downloaded to {result['output']}")


@util.command("unzip")
@click.argument("file_path")
@click.argument("output_path", required=False, default=".")
def util_unzip(file_path: str, output_path: str) -> None:
    """Extract a ZIP archive to the specified output directory."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    folder = os.getcwd() if output_path == "." else output_path
    result = run_operation("util.unzip", file=file_path, output_dir=folder)
    if emit_result(result):
        return
    print_success(f"Extracted to {result['output']}")


@util.command("zip")
@click.argument("source_path")
@click.argument("output_file")
def util_zip(source_path: str, output_file: str) -> None:
    """Create a ZIP archive from a file or directory."""
    from bioamla.cli.console import print_success
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    result = run_operation("util.zip", source=source_path, output=output_file)
    if emit_result(result):
        return
    print_success(f"Created {output_file}")
