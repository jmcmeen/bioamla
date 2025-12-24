"""Configuration management commands."""

import click

from bioamla.core.config import get_config


@click.group()
def config():
    """Configuration management commands."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    from bioamla.cli.progress import console

    config_obj = get_config()

    console.print("\n[bold]Current Configuration[/bold]")
    if config_obj._source:
        console.print(f"[dim]Source: {config_obj._source}[/dim]\n")
    else:
        console.print("[dim]Source: defaults (no config file found)[/dim]\n")

    sections = [
        "project",
        "audio",
        "visualize",
        "models",
        "inference",
        "training",
        "analysis",
        "batch",
        "output",
        "progress",
        "logging",
    ]
    for section_name in sections:
        section = getattr(config_obj, section_name, {})
        if section:
            console.print(f"[bold blue]\\[{section_name}][/bold blue]")
            for key, value in section.items():
                console.print(f"  {key} = {value}")
            console.print()


@config.command("init")
@click.option("--output", "-o", default="bioamla.toml", help="Output file path")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file")
def config_init(output, force):
    """Create a default configuration file."""
    from pathlib import Path

    from bioamla.core.config import create_default_config_file
    from bioamla.cli.progress import print_error, print_success

    path = Path(output)
    if path.exists() and not force:
        print_error(f"File already exists: {output}")
        click.echo("Use --force to overwrite.")
        raise SystemExit(1)

    create_default_config_file(output)
    print_success(f"Created configuration file: {output}")


@config.command("path")
def config_path():
    """Show configuration file search paths."""
    from bioamla.core.config import CONFIG_LOCATIONS, find_config_file
    from bioamla.cli.progress import console

    console.print("\n[bold]Configuration File Search Paths[/bold]\n")
    console.print("Files are searched in order (first found wins):\n")

    active_config = find_config_file()

    for i, location in enumerate(CONFIG_LOCATIONS, 1):
        exists = location.exists()
        status = (
            "[green]✓ ACTIVE[/green]"
            if location == active_config
            else ("[dim]exists[/dim]" if exists else "[dim]not found[/dim]")
        )
        console.print(f"  {i}. {location} {status}")

    console.print()


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


@config.command("purge")
@click.option("--models", is_flag=True, help="Purge cached models")
@click.option("--datasets", is_flag=True, help="Purge cached datasets")
@click.option(
    "--all", "purge_all", is_flag=True, help="Purge all cached data (models and datasets)"
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def config_purge(models: bool, datasets: bool, purge_all: bool, yes: bool):
    """Purge cached HuggingFace Hub data from local storage.

    Examples:
        bioamla config purge --models
        bioamla config purge --datasets
        bioamla config purge --all -y
    """
    import shutil
    from pathlib import Path

    from huggingface_hub import scan_cache_dir

    if not models and not datasets and not purge_all:
        click.echo("Please specify what to purge: --models, --datasets, or --all")
        click.echo("Run 'bioamla config purge --help' for more information.")
        return

    if purge_all:
        models = True
        datasets = True

    cache_info = scan_cache_dir()

    models_to_delete = []
    datasets_to_delete = []

    for repo in cache_info.repos:
        if repo.repo_type == "model" and models:
            models_to_delete.append(repo)
        elif repo.repo_type == "dataset" and datasets:
            datasets_to_delete.append(repo)

    models_size = sum(repo.size_on_disk for repo in models_to_delete)
    datasets_size = sum(repo.size_on_disk for repo in datasets_to_delete)
    total_size = models_size + datasets_size

    if not models_to_delete and not datasets_to_delete:
        click.echo("No cached data found to purge.")
        return

    click.echo("The following cached data will be purged:")
    click.echo()

    if models_to_delete:
        click.echo(f"Models ({len(models_to_delete)} items, {_format_size(models_size)}):")
        for repo in models_to_delete:
            click.echo(f"  - {repo.repo_id} ({_format_size(repo.size_on_disk)})")
        click.echo()

    if datasets_to_delete:
        click.echo(f"Datasets ({len(datasets_to_delete)} items, {_format_size(datasets_size)}):")
        for repo in datasets_to_delete:
            click.echo(f"  - {repo.repo_id} ({_format_size(repo.size_on_disk)})")
        click.echo()

    click.echo(f"Total space to be freed: {_format_size(total_size)}")
    click.echo()

    if not yes:
        if not click.confirm("Are you sure you want to delete this cached data?"):
            click.echo("Aborted.")
            return

    deleted_count = 0
    freed_space = 0

    from huggingface_hub import constants

    cache_path = Path(constants.HF_HUB_CACHE)

    for repo in models_to_delete + datasets_to_delete:
        try:
            for revision in repo.revisions:
                shutil.rmtree(revision.snapshot_path, ignore_errors=True)
            repo_path = cache_path / f"{repo.repo_type}s--{repo.repo_id.replace('/', '--')}"
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            deleted_count += 1
            freed_space += repo.size_on_disk
        except Exception as e:
            click.echo(f"Warning: Failed to delete {repo.repo_id}: {e}")

    click.echo(f"Successfully purged {deleted_count} items, freed {_format_size(freed_space)}.")


@config.command("deps")
@click.option("--install", "do_install", is_flag=True, help="Install missing system dependencies")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def config_deps(do_install: bool, yes: bool):
    """Check or install system dependencies (FFmpeg, libsndfile, PortAudio).

    These system libraries are required for full bioamla functionality:

    \b
      - FFmpeg: Audio format conversion (MP3, FLAC, etc.)
      - libsndfile: Audio file I/O
      - PortAudio: Real-time audio recording

    \b
    Examples:
        bioamla config deps              # Check dependencies
        bioamla config deps --install    # Install missing dependencies
        bioamla config deps --install -y # Install without confirmation
    """
    from bioamla.core.deps import (
        check_all_dependencies,
        detect_os,
        get_full_install_command,
        run_install,
    )
    from bioamla.cli.progress import console

    os_type = detect_os()
    deps = check_all_dependencies()

    console.print("\n[bold]System Dependencies[/bold]")
    console.print(f"[dim]Detected OS: {os_type}[/dim]\n")

    all_installed = True
    missing_deps = []

    for dep in deps:
        if dep.installed:
            version_str = f" (v{dep.version})" if dep.version else ""
            console.print(f"[green]✓[/green] {dep.name}{version_str}")
            console.print(f"  [dim]{dep.description} - {dep.required_for}[/dim]")
        else:
            all_installed = False
            missing_deps.append(dep)
            console.print(f"[red]✗[/red] {dep.name} [red]not installed[/red]")
            console.print(f"  [dim]{dep.description} - {dep.required_for}[/dim]")
            if dep.install_hint:
                console.print(f"  [yellow]Install: {dep.install_hint}[/yellow]")

    console.print()

    if all_installed:
        console.print("[green]All system dependencies are installed![/green]")
        return

    full_command = get_full_install_command(os_type)
    if full_command and not do_install:
        console.print("[bold]To install all missing dependencies:[/bold]")
        console.print(f"  {full_command}")
        console.print()
        console.print("[dim]Or run: bioamla config deps --install[/dim]")

    if do_install:
        console.print()
        if not yes:
            if not click.confirm("Install missing system dependencies?"):
                click.echo("Aborted.")
                return

        console.print("\n[bold]Installing dependencies...[/bold]")
        success, message = run_install(os_type)

        if success:
            console.print(f"[green]{message}[/green]")
            console.print("\n[bold]Verifying installation...[/bold]")
            deps = check_all_dependencies()
            for dep in deps:
                if dep.installed:
                    console.print(f"[green]✓[/green] {dep.name}")
                else:
                    console.print(f"[red]✗[/red] {dep.name}")
        else:
            console.print(f"[red]{message}[/red]")
            raise SystemExit(1)
