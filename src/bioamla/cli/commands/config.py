"""Configuration management commands."""

import click


@click.group()
def config() -> None:
    """Configuration and system information commands."""
    pass


@config.command("version")
def config_version() -> None:
    """Show bioamla version and environment information."""
    from bioamla.cli.progress import console
    from bioamla.cli.service_helpers import handle_result, services

    version_data = handle_result(services.util.get_version())

    console.print("\n[bold]BioAMLA Version Information[/bold]\n")
    console.print(f"  bioamla:  {version_data.bioamla_version}")
    console.print(f"  Python:   {version_data.python_version.split()[0]}")
    console.print(f"  Platform: {version_data.platform}")

    if version_data.pytorch_version:
        console.print(f"  PyTorch:  {version_data.pytorch_version}")
    if version_data.cuda_version:
        console.print(f"  CUDA:     {version_data.cuda_version}")

    console.print()


@config.command("devices")
def config_devices() -> None:
    """Show available compute devices (GPU, MPS, CPU)."""
    from bioamla.cli.progress import console
    from bioamla.cli.service_helpers import handle_result, services

    devices_data = handle_result(services.util.get_device_info())

    console.print("\n[bold]Available Compute Devices[/bold]\n")

    for device in devices_data.devices:
        if device.device_type == "cuda":
            memory_str = f" ({device.memory_gb} GB)" if device.memory_gb else ""
            console.print(f"  [green]✓[/green] {device.name}{memory_str}")
            console.print(f"    [dim]Device ID: {device.device_id}[/dim]")
        elif device.device_type == "mps":
            console.print(f"  [green]✓[/green] {device.name}")
            console.print(f"    [dim]Device ID: {device.device_id}[/dim]")
        else:
            console.print(f"  [dim]• {device.name}[/dim]")
            console.print(f"    [dim]Device ID: {device.device_id}[/dim]")

    console.print()

    # Summary
    if devices_data.cuda_available:
        cuda_count = sum(1 for d in devices_data.devices if d.device_type == "cuda")
        console.print(f"[green]CUDA available[/green] ({cuda_count} GPU{'s' if cuda_count > 1 else ''})")
    elif devices_data.mps_available:
        console.print("[green]Apple MPS available[/green]")
    else:
        console.print("[yellow]No GPU acceleration available - using CPU[/yellow]")

    console.print()


@config.command("show")
def config_show() -> None:
    """Show current configuration."""
    from bioamla.cli.progress import console
    from bioamla.cli.service_helpers import handle_result, services

    config_obj = handle_result(services.config.get_config())

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
def config_init(output: str, force: bool) -> None:
    """Create a default configuration file."""
    from bioamla.cli.progress import print_error, print_success
    from bioamla.cli.service_helpers import services

    result = services.config.create_default_config(output, force=force)

    if not result.success:
        print_error(result.error)
        if "already exists" in result.error:
            click.echo("Use --force to overwrite.")
        raise SystemExit(1)

    print_success(f"Created configuration file: {output}")


@config.command("path")
def config_path() -> None:
    """Show configuration file search paths."""
    from pathlib import Path

    from bioamla.cli.progress import console
    from bioamla.cli.service_helpers import services

    console.print("\n[bold]Configuration File Search Paths[/bold]\n")
    console.print("Files are searched in order (first found wins):\n")

    # Get active config
    active_result = services.config.find_config_file()
    active_config = Path(active_result.data) if active_result.success and active_result.data else None

    # Get all locations
    locations_result = services.config.get_config_locations()
    if locations_result.success:
        locations = [Path(loc) for loc in locations_result.data]
        for i, location in enumerate(locations, 1):
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
def config_purge(models: bool, datasets: bool, purge_all: bool, yes: bool) -> None:
    """Purge cached HuggingFace Hub data from local storage.

    Examples:
        bioamla config purge --models
        bioamla config purge --datasets
        bioamla config purge --all -y
    """
    import shutil
    from pathlib import Path

    from huggingface_hub import constants, scan_cache_dir

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
def config_deps(do_install: bool, yes: bool) -> None:
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
    from bioamla.cli.progress import console
    from bioamla.cli.service_helpers import handle_result, services

    # Get OS type
    os_result = services.dependency.detect_os()
    os_type = os_result.data if os_result.success else "unknown"

    # Check all dependencies
    report = handle_result(services.dependency.check_all())

    console.print("\n[bold]System Dependencies[/bold]")
    console.print(f"[dim]Detected OS: {os_type}[/dim]\n")

    for dep in report.dependencies:
        if dep.installed:
            version_str = f" (v{dep.version})" if dep.version else ""
            console.print(f"[green]✓[/green] {dep.name}{version_str}")
            console.print(f"  [dim]{dep.description} - {dep.required_for}[/dim]")
        else:
            console.print(f"[red]✗[/red] {dep.name} [red]not installed[/red]")
            console.print(f"  [dim]{dep.description} - {dep.required_for}[/dim]")
            if dep.install_hint:
                console.print(f"  [yellow]Install: {dep.install_hint}[/yellow]")

    console.print()

    if report.all_installed:
        console.print("[green]All system dependencies are installed![/green]")
        return

    if report.install_command and not do_install:
        console.print("[bold]To install all missing dependencies:[/bold]")
        console.print(f"  {report.install_command}")
        console.print()
        console.print("[dim]Or run: bioamla config deps --install[/dim]")

    if do_install:
        console.print()
        if not yes:
            if not click.confirm("Install missing system dependencies?"):
                click.echo("Aborted.")
                return

        console.print("\n[bold]Installing dependencies...[/bold]")
        install_result = services.dependency.install(os_type)

        if install_result.success:
            console.print(f"[green]{install_result.message}[/green]")
            console.print("\n[bold]Verifying installation...[/bold]")
            verify_result = services.dependency.check_all()
            if verify_result.success:
                for dep in verify_result.data.dependencies:
                    if dep.installed:
                        console.print(f"[green]✓[/green] {dep.name}")
                    else:
                        console.print(f"[red]✗[/red] {dep.name}")
        else:
            console.print(f"[red]{install_result.error}[/red]")
            raise SystemExit(1)
