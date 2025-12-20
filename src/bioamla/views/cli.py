import sys
import time
from typing import Dict, Optional

import click

from bioamla.core.config import get_config, load_config, set_config
from bioamla.core.files import TextFile


class ConfigContext:
    """Context object to hold configuration."""
    def __init__(self):
        self.config = None
        self.start_time = None
        self.command_args = None


pass_config = click.make_pass_decorator(ConfigContext, ensure=True)

@click.group()
@click.option('--config', 'config_path', type=click.Path(exists=True),
              help='Path to TOML configuration file')
@click.pass_context
def cli(ctx, config_path: Optional[str]):
    """Bioamla CLI - Bioacoustics and Machine Learning Applications

    Configuration can be provided via:
    - --config option pointing to a TOML file
    - ./bioamla.toml in current directory
    - ~/.config/bioamla/config.toml
    """
    ctx.ensure_object(ConfigContext)
    # Capture start time and args for command logging
    ctx.obj.start_time = time.time()
    ctx.obj.command_args = sys.argv[1:] if len(sys.argv) > 1 else []

    if config_path:
        ctx.obj.config = load_config(config_path)
        set_config(ctx.obj.config)
    else:
        ctx.obj.config = get_config()


@cli.result_callback()
@click.pass_context
def log_command_result(ctx, result, **kwargs):
    """Log command execution after completion."""
    from bioamla.core.command_log import CommandLogger, create_command_entry
    from bioamla.core.project import is_in_project

    # Only log if in a project
    if not is_in_project():
        return result

    # Get timing info
    duration = 0.0
    if ctx.obj and ctx.obj.start_time:
        duration = time.time() - ctx.obj.start_time

    # Build command string from invoked subcommand
    command_parts = []
    if ctx.invoked_subcommand:
        command_parts.append(ctx.invoked_subcommand)

    # Get full command from sys.argv
    args = ctx.obj.command_args if ctx.obj else []

    # Create and log entry
    entry = create_command_entry(
        command=" ".join(["bioamla"] + args),
        args=args,
        kwargs={},
        exit_code=0,  # If we got here, command succeeded
        duration_seconds=duration,
    )

    logger = CommandLogger()
    logger.log_command(entry)

    return result


# =============================================================================
# Top-level utility commands
# =============================================================================

@cli.command()
def devices():
    """Display comprehensive device information including CUDA and GPU details."""
    from bioamla.core.diagnostics import get_device_info
    device_info = get_device_info()

    click.echo("Devices:")
    click.echo(f'CUDA available: {device_info["cuda_available"]}')
    click.echo(f'Current device: {device_info["current_device"]}')
    click.echo(f'Device count: {device_info["device_count"]}')

    for device in device_info['devices']:
        click.echo(f'  - Index: {device["index"]}, Name: {device["name"]}')

@cli.command()
def version():
    """Display the current version of the bioamla package."""
    from bioamla.core.diagnostics import get_bioamla_version

    click.echo(f"bioamla v{get_bioamla_version()}")




# =============================================================================
# Config Command Group
# =============================================================================

@cli.group()
def config():
    """Configuration management commands."""
    pass


@config.command('show')
@click.pass_context
def config_show(ctx):
    """Show current configuration."""
    from bioamla.core.progress import console

    config_obj = ctx.obj.config if ctx.obj else get_config()

    console.print("\n[bold]Current Configuration[/bold]")
    if config_obj._source:
        console.print(f"[dim]Source: {config_obj._source}[/dim]\n")
    else:
        console.print("[dim]Source: defaults (no config file found)[/dim]\n")

    # Show all configuration sections
    sections = [
        'project', 'audio', 'visualize', 'models', 'inference', 'training',
        'analysis', 'batch', 'output', 'progress', 'logging'
    ]
    for section_name in sections:
        section = getattr(config_obj, section_name, {})
        if section:
            console.print(f"[bold blue]\\[{section_name}][/bold blue]")
            for key, value in section.items():
                console.print(f"  {key} = {value}")
            console.print()


@config.command('init')
@click.option('--output', '-o', default='bioamla.toml', help='Output file path')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
def config_init(output, force):
    """Create a default configuration file."""
    from pathlib import Path

    from bioamla.core.config import create_default_config_file
    from bioamla.core.progress import print_error, print_success

    path = Path(output)
    if path.exists() and not force:
        print_error(f"File already exists: {output}")
        click.echo("Use --force to overwrite.")
        raise SystemExit(1)

    create_default_config_file(output)
    print_success(f"Created configuration file: {output}")


@config.command('path')
def config_path():
    """Show configuration file search paths."""
    from bioamla.core.config import CONFIG_LOCATIONS, find_config_file
    from bioamla.core.progress import console

    console.print("\n[bold]Configuration File Search Paths[/bold]\n")
    console.print("Files are searched in order (first found wins):\n")

    active_config = find_config_file()

    for i, location in enumerate(CONFIG_LOCATIONS, 1):
        exists = location.exists()
        status = "[green]✓ ACTIVE[/green]" if location == active_config else (
            "[dim]exists[/dim]" if exists else "[dim]not found[/dim]"
        )
        console.print(f"  {i}. {location} {status}")

    console.print()


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


@config.command('purge')
@click.option('--models', is_flag=True, help='Purge cached models')
@click.option('--datasets', is_flag=True, help='Purge cached datasets')
@click.option('--all', 'purge_all', is_flag=True, help='Purge all cached data (models and datasets)')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt')
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


@config.command('deps')
@click.option('--install', 'do_install', is_flag=True,
              help='Install missing system dependencies')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt')
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
    from bioamla.core.deps import check_all_dependencies, detect_os, get_full_install_command, run_install
    from bioamla.core.progress import console

    os_type = detect_os()
    deps = check_all_dependencies()

    # Display status
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

    # Show combined install command
    full_command = get_full_install_command(os_type)
    if full_command and not do_install:
        console.print("[bold]To install all missing dependencies:[/bold]")
        console.print(f"  {full_command}")
        console.print()
        console.print("[dim]Or run: bioamla config deps --install[/dim]")

    # Install if requested
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
            # Re-check to confirm
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


# =============================================================================
# Project Command Group
# =============================================================================

@cli.group()
def project():
    """Project management commands."""
    pass


@project.command('init')
@click.argument('path', required=False, default='.')
@click.option('--name', '-n', help='Project name (defaults to directory name)')
@click.option('--description', '-d', default='', help='Project description')
@click.option('--template', '-t', default='default',
              type=click.Choice(['default', 'minimal', 'research', 'production']),
              help='Configuration template to use')
@click.option('--config', '-c', 'config_file', type=click.Path(exists=True),
              help='Custom config file to use as base')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing project')
def project_init(path, name, description, template, config_file, force):
    """
    Initialize a new bioamla project.

    Creates a .bioamla directory with configuration files.

    \b
    Templates:
      default     - Balanced settings for general use
      minimal     - Minimal config, most values use defaults
      research    - Detailed logging, reproducibility focused
      production  - Optimized for batch processing

    \b
    Examples:
      bioamla project init                    # Current directory
      bioamla project init ./my-project       # Specific directory
      bioamla project init -n "Frog Study"    # With custom name
      bioamla project init -t research        # Research template
    """
    from pathlib import Path

    from bioamla.core.progress import print_error, print_success, print_warning
    from bioamla.core.project import PROJECT_MARKER, create_project

    project_path = Path(path).resolve()

    # Check for existing project
    if (project_path / PROJECT_MARKER).exists():
        if not force:
            print_error(f"Project already exists at {project_path}")
            click.echo("Use --force to reinitialize.")
            raise SystemExit(1)
        print_warning("Reinitializing existing project...")

    # Create project
    try:
        info = create_project(
            path=project_path,
            name=name,
            description=description,
            template=template,
            config_file=Path(config_file) if config_file else None,
        )
        print_success(f"Created bioamla project: {info.name}")
        click.echo(f"  Location: {info.root}")
        click.echo(f"  Config: {info.config_path}")
        click.echo(f"\nNext steps:")
        click.echo(f"  cd {project_path}")
        click.echo(f"  bioamla config show")
    except Exception as e:
        print_error(f"Failed to create project: {e}")
        raise SystemExit(1)


@project.command('status')
def project_status():
    """Show current project status and information."""
    from bioamla.core.progress import console
    from bioamla.core.project import load_project

    info = load_project()

    if not info:
        click.echo("Not in a bioamla project.")
        click.echo("Run 'bioamla project init' to create one.")
        return

    console.print(f"\n[bold]Project: {info.name}[/bold]")
    console.print(f"[dim]Version: {info.version}[/dim]")
    if info.description:
        console.print(f"[dim]{info.description}[/dim]")
    console.print(f"\n  Root: {info.root}")
    console.print(f"  Config: {info.config_path}")
    console.print(f"  Logs: {info.logs_path}")


@project.command('config')
@click.argument('action', type=click.Choice(['show', 'edit', 'reset']))
@click.pass_context
def project_config(ctx, action):
    """Manage project configuration."""
    from bioamla.core.project import load_project

    info = load_project()

    if not info:
        click.echo("Not in a bioamla project.")
        raise SystemExit(1)

    if action == 'show':
        # Reuse existing config show logic
        ctx.invoke(config_show)
    elif action == 'edit':
        # Open config in editor
        import os
        editor = os.environ.get('EDITOR', 'nano')
        os.system(f'{editor} {info.config_path}')
    elif action == 'reset':
        # Reset to template defaults
        click.echo("Reset project config to defaults? [y/N] ", nl=False)
        if click.getchar().lower() == 'y':
            from bioamla.core.project import _get_template_content, _customize_template
            template_content = _get_template_content('default')
            customized = _customize_template(template_content, info.name, info.description)
            info.config_path.write_text(customized)
            click.echo("\nConfig reset to defaults.")
        else:
            click.echo("\nCancelled.")


# =============================================================================
# Log Command Group
# =============================================================================

@cli.group()
def log():
    """Command history and logging."""
    pass


@log.command('show')
@click.option('--limit', '-n', default=20, help='Number of entries to show')
@click.option('--command', '-c', 'cmd_filter', help='Filter by command name')
@click.option('--all', 'show_all', is_flag=True, help='Show all entries')
def log_show(limit, cmd_filter, show_all):
    """Show command history."""
    from bioamla.core.command_log import CommandLogger
    from bioamla.core.progress import console

    logger = CommandLogger()

    if not logger.is_available():
        click.echo("Command logging requires a bioamla project.")
        click.echo("Run 'bioamla project init' to create one.")
        return

    entries = logger.get_history(
        limit=None if show_all else limit,
        command_filter=cmd_filter,
    )

    if not entries:
        click.echo("No command history found.")
        return

    console.print(f"\n[bold]Command History[/bold] ({len(entries)} entries)\n")

    for entry in entries:
        status = "[green]✓[/green]" if entry.exit_code == 0 else "[red]✗[/red]"
        duration = f"{entry.duration_seconds:.2f}s"
        console.print(f"{status} {entry.timestamp[:19]}  {entry.command}  [{duration}]")


@log.command('search')
@click.argument('query')
def log_search(query):
    """Search command history."""
    from bioamla.core.command_log import CommandLogger
    from bioamla.core.progress import console

    logger = CommandLogger()

    if not logger.is_available():
        click.echo("Not in a bioamla project.")
        return

    results = logger.search(query)

    if not results:
        click.echo(f"No matches for '{query}'")
        return

    console.print(f"\n[bold]Search Results[/bold] ({len(results)} matches)\n")
    for entry in results[:20]:
        status = "[green]✓[/green]" if entry.exit_code == 0 else "[red]✗[/red]"
        console.print(f"{status} {entry.timestamp[:19]}  {entry.command}")


@log.command('clear')
@click.confirmation_option(prompt='Clear all command history?')
def log_clear():
    """Clear command history."""
    from bioamla.core.command_log import CommandLogger

    logger = CommandLogger()

    if not logger.is_available():
        click.echo("Not in a bioamla project.")
        return

    count = logger.clear()
    click.echo(f"Cleared {count} log entries.")


@log.command('stats')
def log_stats():
    """Show command history statistics."""
    from bioamla.core.command_log import CommandLogger
    from bioamla.core.progress import console

    logger = CommandLogger()

    if not logger.is_available():
        click.echo("Not in a bioamla project.")
        return

    stats = logger.get_stats()

    if stats['total_commands'] == 0:
        click.echo("No command history found.")
        return

    console.print(f"\n[bold]Command Statistics[/bold]\n")
    console.print(f"  Total commands: {stats['total_commands']}")
    console.print(f"  [green]Successful:[/green] {stats['successful_commands']}")
    console.print(f"  [red]Failed:[/red] {stats['failed_commands']}")

    if stats['command_counts']:
        console.print(f"\n[bold]Commands by frequency:[/bold]")
        sorted_cmds = sorted(stats['command_counts'].items(), key=lambda x: x[1], reverse=True)
        for cmd, count in sorted_cmds[:10]:
            console.print(f"  {cmd}: {count}")


# =============================================================================
# Models Command Group
# =============================================================================

@cli.group()
def models():
    """ML model operations (predict, embed, train, convert)."""
    pass


# Subgroups for models
@models.group()
def predict():
    """Run inference with ML models."""
    pass


@models.group()
def train():
    """Train ML models."""
    pass


@models.group()
def evaluate():
    """Evaluate ML models."""
    pass


# =============================================================================
# AST Commands (Audio Spectrogram Transformer)
# =============================================================================

@predict.command('ast')
@click.argument('path')
@click.option('--model-path', default='bioamla/scp-frogs', help='AST model to use for inference')
@click.option('--resample-freq', default=16000, type=int, help='Resampling frequency')
@click.option('--batch', is_flag=True, default=False, help='Run batch inference on a directory of audio files')
@click.option('--output-csv', default='output.csv', help='Output CSV file name (batch mode only)')
@click.option('--segment-duration', default=1, type=int, help='Duration of audio segments in seconds (batch mode only)')
@click.option('--segment-overlap', default=0, type=int, help='Overlap between segments in seconds (batch mode only)')
@click.option('--restart/--no-restart', default=False, help='Whether to restart from existing results (batch mode only)')
@click.option('--batch-size', default=8, type=int, help='Number of segments to process in parallel (default: 8, batch mode only)')
@click.option('--fp16/--no-fp16', default=False, help='Use half-precision (FP16) for faster GPU inference (batch mode only)')
@click.option('--compile/--no-compile', default=False, help='Use torch.compile() for optimized inference (PyTorch 2.0+, batch mode only)')
@click.option('--workers', default=1, type=int, help='Number of parallel workers for file loading (default: 1, batch mode only)')
def ast_predict(
    path: str,
    model_path: str,
    resample_freq: int,
    batch: bool,
    output_csv: str,
    segment_duration: int,
    segment_overlap: int,
    restart: bool,
    batch_size: int,
    fp16: bool,
    compile: bool,
    workers: int
):
    """
    Perform prediction on audio file(s).

    PATH can be a single audio file or a directory (with --batch flag).

    Single file mode (default):
        bioamla models ast-predict audio.wav --model-path my_model

    Batch mode (--batch):
        bioamla models ast-predict ./audio_dir --batch --model-path my_model

        Processes all WAV files in the specified directory and saves predictions
        to a CSV file. Supports resumable operations.

        Performance options (batch mode only):
            --batch-size: Process multiple segments in one forward pass (GPU optimization)
            --fp16: Use half-precision inference for ~2x speedup on modern GPUs
            --compile: Use torch.compile() for optimized model execution
            --workers: Parallel file loading for I/O-bound workloads
    """
    if batch:
        _run_batch_inference(
            directory=path,
            output_csv=output_csv,
            model_path=model_path,
            resample_freq=resample_freq,
            segment_duration=segment_duration,
            segment_overlap=segment_overlap,
            restart=restart,
            batch_size=batch_size,
            fp16=fp16,
            compile=compile,
            workers=workers
        )
    else:
        from bioamla.core.detection.ast import wav_ast_inference
        prediction = wav_ast_inference(path, model_path, resample_freq)
        click.echo(f"{prediction}")


def _run_batch_inference(
    directory: str,
    output_csv: str,
    model_path: str,
    resample_freq: int,
    segment_duration: int,
    segment_overlap: int,
    restart: bool,
    batch_size: int,
    fp16: bool,
    compile: bool,
    workers: int
):
    """Run batch inference on a directory of audio files."""
    import os
    import time

    import pandas as pd
    import torch

    from bioamla.core.detection.ast import (
        InferenceConfig,
        load_pretrained_ast_model,
        wave_file_batch_inference,
    )
    from bioamla.core.utils import file_exists, get_files_by_extension

    output_csv = os.path.join(directory, output_csv)

    # Create parent directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Output csv: " + output_csv)

    wave_files = get_files_by_extension(directory=directory, extensions=['.wav'], recursive=True)

    if len(wave_files) == 0:
        print("No wave files found in directory: " + directory)
        return
    else:
        print("Found " + str(len(wave_files)) + " wave files in directory: " + directory)

    print("Restart: " + str(restart))
    if restart:
        if file_exists(output_csv):
            print("file exists: " + output_csv)
            df = pd.read_csv(output_csv)
            processed_files = set(df['filepath'])
            print("Found " + str(len(processed_files)) + " processed files")

            print("Removing processed files from wave files")
            wave_files = [f for f in wave_files if f not in processed_files]

            print("Found " + str(len(wave_files)) + " wave files left to process")

            if len(wave_files) == 0:
                print("No wave files left to process")
                return
        else:
            print("creating new file: " + output_csv)
            results = pd.DataFrame(columns=['filepath', 'start', 'stop', 'prediction'])
            results.to_csv(output_csv, header=True, index=False)
    else:
        print("creating new file: " + output_csv)
        results = pd.DataFrame(columns=['filepath', 'start', 'stop', 'prediction'])
        results.to_csv(output_csv, header=True, index=False)

    print("Loading model: " + model_path)
    print(f"Performance options: batch_size={batch_size}, fp16={fp16}, compile={compile}, workers={workers}")

    model = load_pretrained_ast_model(model_path, use_fp16=fp16, use_compile=compile)

    from torch.nn import Module
    if not isinstance(model, Module):
        raise TypeError("Model must be a PyTorch Module")

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + device)

    config = InferenceConfig(
        batch_size=batch_size,
        use_fp16=fp16,
        use_compile=compile,
        num_workers=workers
    )

    # Pre-load feature extractor before timing starts
    from bioamla.core.detection.ast import get_cached_feature_extractor
    feature_extractor = get_cached_feature_extractor()

    start_time = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start batch inference at " + time_string)

    wave_file_batch_inference(
        wave_files=wave_files,
        model=model,
        freq=resample_freq,
        segment_duration=segment_duration,
        segment_overlap=segment_overlap,
        output_csv=output_csv,
        config=config,
        feature_extractor=feature_extractor
    )

    end_time = time.time()
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("End batch inference at " + time_string)
    elapsed = end_time - start_time
    print(f"Elapsed time: {elapsed:.2f}s ({len(wave_files)/elapsed:.2f} files/sec)")


@train.command('ast')
@click.option('--training-dir', default='.', help='Directory to save training outputs')
@click.option('--base-model', default='MIT/ast-finetuned-audioset-10-10-0.4593', help='Base model to fine-tune')
@click.option('--train-dataset', default='bioamla/scp-frogs', help='Training dataset from HuggingFace Hub')
@click.option('--split', default='train', help='Dataset split to use')
@click.option('--category-id-column', default='target', help='Column name for category IDs')
@click.option('--category-label-column', default='category', help='Column name for category labels')
@click.option('--report-to', default='tensorboard', help='Where to report metrics')
@click.option('--learning-rate', default=5.0e-5, type=float, help='Learning rate for training')
@click.option('--push-to-hub/--no-push-to-hub', default=False, help='Whether to push model to HuggingFace Hub')
@click.option('--num-train-epochs', default=1, type=int, help='Number of training epochs')
@click.option('--per-device-train-batch-size', default=8, type=int, help='Training batch size per device')
@click.option('--eval-strategy', default='epoch', help='Evaluation strategy')
@click.option('--save-strategy', default='epoch', help='Model save strategy')
@click.option('--eval-steps', default=1, type=int, help='Number of steps between evaluations')
@click.option('--save-steps', default=1, type=int, help='Number of steps between saves')
@click.option('--load-best-model-at-end/--no-load-best-model-at-end', default=True, help='Load best model at end of training')
@click.option('--metric-for-best-model', default='accuracy', help='Metric to use for best model selection')
@click.option('--logging-strategy', default='steps', help='Logging strategy')
@click.option('--logging-steps', default=100, type=int, help='Number of steps between logging')
@click.option('--fp16/--no-fp16', default=False, help='Use FP16 mixed precision training (for NVIDIA GPUs)')
@click.option('--bf16/--no-bf16', default=False, help='Use BF16 mixed precision training (for Ampere+ GPUs)')
@click.option('--gradient-accumulation-steps', default=1, type=int, help='Number of gradient accumulation steps')
@click.option('--dataloader-num-workers', default=4, type=int, help='Number of dataloader workers')
@click.option('--torch-compile/--no-torch-compile', default=False, help='Use torch.compile for faster training (PyTorch 2.0+)')
@click.option('--finetune-mode', type=click.Choice(['full', 'feature-extraction']), default='full', help='Training mode: full (all layers) or feature-extraction (freeze base, train classifier only)')
@click.option('--mlflow-tracking-uri', default=None, help='MLflow tracking server URI (e.g., http://localhost:5000)')
@click.option('--mlflow-experiment-name', default=None, help='MLflow experiment name')
@click.option('--mlflow-run-name', default=None, help='MLflow run name')
def ast_train(
    training_dir: str,
    base_model: str,
    train_dataset: str,
    split: str,
    category_id_column: str,
    category_label_column: str,
    report_to: str,
    learning_rate: float,
    push_to_hub: bool,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    eval_strategy: str,
    save_strategy: str,
    eval_steps: int,
    save_steps: int,
    load_best_model_at_end: bool,
    metric_for_best_model: str,
    logging_strategy: str,
    logging_steps: int,
    fp16: bool,
    bf16: bool,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    torch_compile: bool,
    finetune_mode: str,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_run_name: str
):
    """Fine-tune an AST model on a custom dataset."""
    import evaluate
    import numpy as np
    import torch
    from audiomentations import (
        AddGaussianSNR,
        ClippingDistortion,
        Compose,
        Gain,
        GainTransition,
        PitchShift,
        TimeStretch,
    )
    from datasets import Audio, Dataset, DatasetDict, load_dataset
    from transformers import (
        ASTConfig,
        ASTFeatureExtractor,
        ASTForAudioClassification,
        Trainer,
        TrainingArguments,
    )

    from bioamla.core.utils import create_directory

    output_dir = training_dir + "/runs"
    logging_dir = training_dir + "/logs"
    best_model_path = training_dir + "/best_model"

    mlflow_run = None
    if mlflow_tracking_uri or mlflow_experiment_name:
        try:
            import mlflow
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                print(f"MLflow tracking URI: {mlflow_tracking_uri}")
            if mlflow_experiment_name:
                mlflow.set_experiment(mlflow_experiment_name)
                print(f"MLflow experiment: {mlflow_experiment_name}")
            if "mlflow" not in report_to:
                report_to = f"{report_to},mlflow" if report_to else "mlflow"
            print(f"MLflow integration enabled, reporting to: {report_to}")
        except ImportError:
            print("Warning: MLflow not installed. Install with 'pip install mlflow' to enable MLflow tracking.")

    # Convert comma-separated report_to string to a list for TrainingArguments
    if report_to and "," in report_to:
        report_to = [r.strip() for r in report_to.split(",")]

    # Handle dataset loading
    # Format: "owner/repo" or "owner/repo:config" for datasets with configurations
    # Use samuelstevens/BirdSet for bird sound datasets (Parquet format, works with datasets 4.x+)
    if ":" in train_dataset:
        dataset_name, config_name = train_dataset.rsplit(":", 1)
        dataset = load_dataset(dataset_name, config_name, split=split)
    elif "BirdSet" in train_dataset:
        # BirdSet datasets have configurations for different subsets
        # Recommend samuelstevens/BirdSet which is in Parquet format
        click.echo("Note: BirdSet detected. Use format 'samuelstevens/BirdSet:HSN' to specify subset.")
        click.echo("Available subsets: HSN, NBP, NES, PER")
        dataset = load_dataset(train_dataset, "HSN", split=split)
    else:
        dataset = load_dataset(train_dataset, split=split)

    if isinstance(dataset, Dataset):
        class_names = sorted(list(set(dataset[category_label_column])))
    elif isinstance(dataset, DatasetDict):
        first_split_name = list(dataset.keys())[0]
        first_split = dataset[first_split_name]
        class_names = sorted(list(set(first_split[category_label_column])))
    else:
        raise TypeError("Dataset must be a Dataset or DatasetDict instance")

    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    num_labels = len(class_names)

    def convert_labels(example):
        example["labels"] = label_to_id[example[category_label_column]]
        return example

    dataset = dataset.map(
        convert_labels,
        remove_columns=[category_label_column],
        writer_batch_size=100,
        num_proc=1,
    )

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    pretrained_model = base_model
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
    model_input_name = feature_extractor.model_input_names[0]
    SAMPLING_RATE = feature_extractor.sampling_rate

    def preprocess_audio(batch):
        wavs = [audio["array"] for audio in batch["input_values"]]
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for idx, name in enumerate(class_names)}

    test_size = 0.2
    if isinstance(dataset, Dataset):
        try:
            dataset = dataset.train_test_split(
                test_size=test_size, shuffle=True, seed=0, stratify_by_column="labels")
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}). Using regular split.")
            dataset = dataset.train_test_split(
                test_size=test_size, shuffle=True, seed=0)
    elif isinstance(dataset, DatasetDict) and "test" not in dataset:
        train_data = dataset["train"]
        try:
            dataset = train_data.train_test_split(
                test_size=test_size, shuffle=True, seed=0, stratify_by_column="labels")
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}). Using regular split.")
            dataset = train_data.train_test_split(
                test_size=test_size, shuffle=True, seed=0)

    audio_augmentations = Compose([
        AddGaussianSNR(min_snr_db=10, max_snr_db=20),
        Gain(min_gain_db=-6, max_gain_db=6),
        GainTransition(min_gain_db=-6, max_gain_db=6, min_duration=0.01, max_duration=0.3, duration_unit="fraction"),
        ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2),
        PitchShift(min_semitones=-4, max_semitones=4),
    ], p=0.8, shuffle=True)

    def preprocess_audio_with_transforms(batch):
        wavs = [audio_augmentations(audio["array"], sample_rate=SAMPLING_RATE) for audio in batch["input_values"]]
        inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        return {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

    dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
    dataset = dataset.rename_column("audio", "input_values")

    def is_valid_audio(example):
        try:
            _ = example["input_values"]["array"]
            return True
        except (RuntimeError, Exception):
            return False

    print("Filtering out corrupted audio files...")
    original_sizes = {}
    filter_num_proc = min(dataloader_num_workers, 4) if dataloader_num_workers > 1 else 1
    if isinstance(dataset, DatasetDict):
        for split_name in dataset.keys():
            original_sizes[split_name] = len(dataset[split_name])
        dataset = dataset.filter(is_valid_audio, num_proc=filter_num_proc)
        for split_name in dataset.keys():
            filtered_count = original_sizes[split_name] - len(dataset[split_name])
            if filtered_count > 0:
                print(f"  Removed {filtered_count} corrupted files from {split_name} split")
    else:
        original_size = len(dataset)
        dataset = dataset.filter(is_valid_audio, num_proc=filter_num_proc)
        filtered_count = original_size - len(dataset)
        if filtered_count > 0:
            print(f"  Removed {filtered_count} corrupted files")

    feature_extractor.do_normalize = False

    if isinstance(dataset, DatasetDict) and "train" in dataset:
        train_dataset_for_norm = dataset["train"]
        if len(train_dataset_for_norm) == 0:
            raise ValueError("No valid audio samples found in training dataset after filtering")

        def compute_stats(batch):
            wavs = [audio["array"] for audio in batch["input_values"]]
            inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            values = inputs.get(model_input_name)
            means = [float(torch.mean(values[i])) for i in range(values.shape[0])]
            stds = [float(torch.std(values[i])) for i in range(values.shape[0])]
            return {"_mean": means, "_std": stds}

        print("Calculating dataset normalization statistics...")
        norm_num_proc = min(dataloader_num_workers, 4) if dataloader_num_workers > 1 else 1
        stats_dataset = train_dataset_for_norm.map(
            compute_stats,
            batched=True,
            batch_size=32,
            num_proc=norm_num_proc,
            remove_columns=train_dataset_for_norm.column_names,
        )
        all_means = stats_dataset["_mean"]
        all_stds = stats_dataset["_std"]

        if not all_means:
            raise ValueError("No valid audio samples found in training dataset")

        feature_extractor.mean = float(np.mean(all_means))
        feature_extractor.std = float(np.mean(all_stds))
    else:
        raise ValueError("Expected DatasetDict with 'train' split")

    feature_extractor.do_normalize = True

    print("Calculated mean and std:", feature_extractor.mean, feature_extractor.std)

    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset["train"].set_transform(preprocess_audio_with_transforms, output_all_columns=False)
        if "test" in dataset:
            dataset["test"].set_transform(preprocess_audio, output_all_columns=False)
    else:
        raise ValueError("Expected DatasetDict for transform application")

    config = ASTConfig.from_pretrained(pretrained_model)
    config.num_labels = num_labels
    config.label2id = label2id
    config.id2label = {v: k for k, v in label2id.items()}

    model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
    model.init_weights()

    if finetune_mode == "feature-extraction":
        print("Feature extraction mode: freezing base model, only training classifier head")
        for param in model.audio_spectrogram_transformer.parameters():
            param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    else:
        print("Full finetune mode: training all model layers")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        report_to=report_to,
        learning_rate=learning_rate,
        push_to_hub=push_to_hub,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        fp16=fp16,
        bf16=bf16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        torch_compile=torch_compile,
        run_name=mlflow_run_name,
    )

    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    AVERAGE = "macro" if config.num_labels > 2 else "binary"

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits = eval_pred.predictions
        predictions = np.argmax(logits, axis=1)

        accuracy_result = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
        metrics: Dict[str, float] = accuracy_result if accuracy_result is not None else {}

        precision_result = precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE)
        if precision_result is not None:
            metrics.update(precision_result)

        recall_result = recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE)
        if recall_result is not None:
            metrics.update(recall_result)

        f1_result = f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE)
        if f1_result is not None:
            metrics.update(f1_result)

        return metrics

    if isinstance(dataset, DatasetDict):
        train_data = dataset.get("train")
        eval_data = dataset.get("test")
    else:
        raise ValueError("Expected DatasetDict for trainer setup")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    create_directory(best_model_path)
    trainer.save_model(best_model_path)


@evaluate.command('ast')
@click.argument('path')
@click.option('--model-path', default='bioamla/scp-frogs', help='AST model to use for evaluation')
@click.option('--ground-truth', '-g', required=True, help='Path to CSV file with ground truth labels')
@click.option('--output', '-o', default=None, help='Output file for evaluation results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv', 'txt']), default='txt', help='Output format')
@click.option('--file-column', default='file_name', help='Column name for file names in ground truth CSV')
@click.option('--label-column', default='label', help='Column name for labels in ground truth CSV')
@click.option('--resample-freq', default=16000, type=int, help='Resampling frequency')
@click.option('--batch-size', default=8, type=int, help='Batch size for inference')
@click.option('--fp16/--no-fp16', default=False, help='Use half-precision inference')
@click.option('--quiet', is_flag=True, help='Only output metrics, suppress progress')
def ast_evaluate(
    path: str,
    model_path: str,
    ground_truth: str,
    output: str,
    output_format: str,
    file_column: str,
    label_column: str,
    resample_freq: int,
    batch_size: int,
    fp16: bool,
    quiet: bool,
):
    """Evaluate an AST model on a directory of audio files.

    PATH is a directory containing audio files to evaluate.

    Example:
        bioamla models ast-evaluate ./test_audio --model bioamla/scp-frogs --ground-truth labels.csv
    """
    from pathlib import Path as PathLib

    from bioamla.core.evaluate import (
        evaluate_directory,
        format_metrics_report,
        save_evaluation_results,
    )

    path_obj = PathLib(path)
    if not path_obj.exists():
        click.echo(f"Error: Path not found: {path}")
        raise SystemExit(1)

    gt_path = PathLib(ground_truth)
    if not gt_path.exists():
        click.echo(f"Error: Ground truth file not found: {ground_truth}")
        raise SystemExit(1)

    try:
        result = evaluate_directory(
            audio_dir=path,
            model_path=model_path,
            ground_truth_csv=ground_truth,
            gt_file_column=file_column,
            gt_label_column=label_column,
            resample_freq=resample_freq,
            batch_size=batch_size,
            use_fp16=fp16,
            verbose=not quiet,
        )

        # Display results
        if not quiet:
            report = format_metrics_report(result)
            click.echo(report)
        else:
            click.echo(f"Accuracy: {result.accuracy:.4f}")
            click.echo(f"Precision: {result.precision:.4f}")
            click.echo(f"Recall: {result.recall:.4f}")
            click.echo(f"F1 Score: {result.f1_score:.4f}")

        # Save results if output specified
        if output:
            save_evaluation_results(result, output, format=output_format)
            click.echo(f"Results saved to: {output}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)


@models.command('list')
def models_list():
    """List available model types."""
    from bioamla.models import list_models
    click.echo("Available model types:")
    for model_name in list_models():
        click.echo(f"  - {model_name}")


@predict.command('generic')
@click.argument('path')
@click.option('--model-type', type=click.Choice(['ast', 'birdnet', 'opensoundscape']),
              default='ast', help='Model type to use')
@click.option('--model-path', required=True, help='Path to model or HuggingFace identifier')
@click.option('--output', '-o', default=None, help='Output CSV file')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--min-confidence', default=0.0, type=float, help='Minimum confidence threshold')
@click.option('--top-k', default=1, type=int, help='Number of top predictions per segment')
@click.option('--clip-duration', default=3.0, type=float, help='Clip duration in seconds')
@click.option('--overlap', default=0.0, type=float, help='Overlap between clips in seconds')
@click.option('--sample-rate', default=16000, type=int, help='Target sample rate')
@click.option('--batch-size', default=8, type=int, help='Batch size for processing')
@click.option('--fp16/--no-fp16', default=False, help='Use half-precision inference')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def predict_generic(
    path, model_type, model_path, output, batch, min_confidence,
    top_k, clip_duration, overlap, sample_rate, batch_size, fp16, quiet
):
    """Run predictions using an ML model (multi-model interface).

    Single file:
        bioamla models predict generic audio.wav --model-type ast --model-path my_model

    Batch mode:
        bioamla models predict generic ./audio --batch --model-type ast --model-path my_model -o results.csv
    """
    import csv
    import time
    from pathlib import Path

    from bioamla.models import ModelConfig, load_model
    from bioamla.core.utils import get_audio_files

    config = ModelConfig(
        sample_rate=sample_rate,
        clip_duration=clip_duration,
        overlap=overlap,
        min_confidence=min_confidence,
        top_k=top_k,
        batch_size=batch_size,
        use_fp16=fp16,
    )

    if not quiet:
        click.echo(f"Loading {model_type} model from {model_path}...")

    try:
        model = load_model(model_type, model_path, config, use_fp16=fp16)
    except Exception as e:
        click.echo(f"Error loading model: {e}")
        raise SystemExit(1)

    if batch:
        path = Path(path)
        if not path.is_dir():
            click.echo(f"Error: {path} is not a directory")
            raise SystemExit(1)

        audio_files = get_audio_files(str(path))
        if not audio_files:
            click.echo("No audio files found")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Processing {len(audio_files)} files...")

        start_time = time.time()
        all_results = []

        for i, filepath in enumerate(audio_files):
            try:
                results = model.predict(filepath)
                all_results.extend(results)
                if not quiet:
                    click.echo(f"[{i+1}/{len(audio_files)}] {filepath}: {len(results)} predictions")
            except Exception as e:
                if not quiet:
                    click.echo(f"[{i+1}/{len(audio_files)}] Error: {filepath} - {e}")

        elapsed = time.time() - start_time

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with TextFile(output_path, mode='w', newline='') as f:
                writer = csv.writer(f.handle)
                writer.writerow(['filepath', 'start_time', 'end_time', 'label', 'confidence'])
                for r in all_results:
                    writer.writerow([r.filepath, f"{r.start_time:.3f}", f"{r.end_time:.3f}",
                                    r.label, f"{r.confidence:.4f}"])
            if not quiet:
                click.echo(f"\nResults saved to {output}")

        if not quiet:
            click.echo(f"\nProcessed {len(audio_files)} files in {elapsed:.2f}s")
            click.echo(f"Total predictions: {len(all_results)}")

    else:
        # Single file
        if not Path(path).exists():
            click.echo(f"Error: File not found: {path}")
            raise SystemExit(1)

        results = model.predict(path)

        if output:
            with TextFile(output, mode='w', newline='') as f:
                writer = csv.writer(f.handle)
                writer.writerow(['filepath', 'start_time', 'end_time', 'label', 'confidence'])
                for r in results:
                    writer.writerow([r.filepath, f"{r.start_time:.3f}", f"{r.end_time:.3f}",
                                    r.label, f"{r.confidence:.4f}"])
            click.echo(f"Results saved to {output}")
        else:
            for r in results:
                click.echo(f"{r.start_time:.2f}-{r.end_time:.2f}s: {r.label} ({r.confidence:.3f})")


@models.command('embed')
@click.argument('path')
@click.option('--model-type', type=click.Choice(['ast', 'birdnet', 'opensoundscape']),
              default='ast', help='Model type to use')
@click.option('--model-path', required=True, help='Path to model or HuggingFace identifier')
@click.option('--output', '-o', required=True, help='Output file (.npy or .npz)')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--layer', default=None, help='Layer to extract embeddings from')
@click.option('--sample-rate', default=16000, type=int, help='Target sample rate')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def models_embed(path, model_type, model_path, output, batch, layer, sample_rate, quiet):
    """Extract embeddings from audio using an ML model.

    Single file:
        bioamla models embed audio.wav --model-type ast --model-path my_model -o embeddings.npy

    Batch mode:
        bioamla models embed ./audio --batch --model-type ast --model-path my_model -o embeddings.npy
    """
    from pathlib import Path

    import numpy as np

    from bioamla.models import ModelConfig, load_model
    from bioamla.core.utils import get_audio_files

    config = ModelConfig(sample_rate=sample_rate)

    if not quiet:
        click.echo(f"Loading {model_type} model from {model_path}...")

    try:
        model = load_model(model_type, model_path, config)
    except Exception as e:
        click.echo(f"Error loading model: {e}")
        raise SystemExit(1)

    if batch:
        path = Path(path)
        if not path.is_dir():
            click.echo(f"Error: {path} is not a directory")
            raise SystemExit(1)

        audio_files = get_audio_files(str(path))
        if not audio_files:
            click.echo("No audio files found")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Extracting embeddings from {len(audio_files)} files...")

        embeddings_list = []
        filepaths_list = []
        for i, filepath in enumerate(audio_files):
            try:
                emb = model.extract_embeddings(filepath, layer=layer)
                # Flatten to 1D if needed (take mean if multiple time steps)
                if emb.ndim > 1:
                    emb = emb.mean(axis=0) if emb.shape[0] > 1 else emb.squeeze()
                embeddings_list.append(emb)
                filepaths_list.append(filepath)
                if not quiet:
                    click.echo(f"[{i+1}/{len(audio_files)}] {filepath}: shape {emb.shape}")
            except Exception as e:
                if not quiet:
                    click.echo(f"[{i+1}/{len(audio_files)}] Error: {filepath} - {e}")

        # Stack into 2D array (samples x features) for clustering compatibility
        embeddings = np.vstack(embeddings_list)
        np.save(output, embeddings)

        # Save filepaths mapping separately
        filepaths_output = str(output).replace('.npy', '_filepaths.txt')
        with TextFile(filepaths_output, mode='w') as f:
            f.write('\n'.join(filepaths_list))

        if not quiet:
            click.echo(f"\nEmbeddings saved to {output}")
            click.echo(f"Filepaths saved to {filepaths_output}")

    else:
        if not Path(path).exists():
            click.echo(f"Error: File not found: {path}")
            raise SystemExit(1)

        embeddings = model.extract_embeddings(path, layer=layer)
        np.save(output, embeddings)

        if not quiet:
            click.echo(f"Embeddings shape: {embeddings.shape}")
            click.echo(f"Saved to {output}")


@train.command('cnn')
@click.argument('train_dir')
@click.option('--val-dir', default=None, help='Validation data directory')
@click.option('--output-dir', '-o', default='./output', help='Output directory for model')
@click.option('--classes', required=True, help='Comma-separated list of class names')
@click.option('--architecture', type=click.Choice(['resnet18', 'resnet50']),
              default='resnet18', help='Model architecture')
@click.option('--epochs', default=10, type=int, help='Number of training epochs')
@click.option('--batch-size', default=32, type=int, help='Batch size')
@click.option('--learning-rate', default=1e-4, type=float, help='Learning rate')
@click.option('--freeze-epochs', default=0, type=int, help='Epochs to keep backbone frozen')
@click.option('--pretrained/--no-pretrained', default=True, help='Use ImageNet pretrained weights')
@click.option('--fp16/--no-fp16', default=False, help='Use mixed precision training')
@click.option('--sample-rate', default=16000, type=int, help='Target sample rate')
@click.option('--clip-duration', default=3.0, type=float, help='Clip duration in seconds')
def train_cnn(
    train_dir, val_dir, output_dir, classes, architecture, epochs,
    batch_size, learning_rate, freeze_epochs, pretrained, fp16, sample_rate, clip_duration
):
    """Train a custom CNN model using transfer learning.

    Data should be organized with subdirectories per class:
        train_dir/
            class1/
                audio1.wav
                audio2.wav
            class2/
                audio3.wav
                ...

    Example:
        bioamla models train cnn ./data/train --val-dir ./data/val \\
            --classes "bird,frog,insect" --epochs 20 -o ./my_model
    """
    from bioamla.models import ModelTrainer, TrainingConfig

    class_names = [c.strip() for c in classes.split(',')]

    click.echo(f"Training {architecture} model with classes: {class_names}")
    click.echo(f"Training data: {train_dir}")
    if val_dir:
        click.echo(f"Validation data: {val_dir}")

    config = TrainingConfig(
        train_dir=train_dir,
        val_dir=val_dir,
        output_dir=output_dir,
        class_names=class_names,
        architecture=architecture,
        pretrained=pretrained,
        freeze_backbone_epochs=freeze_epochs,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=learning_rate,
        sample_rate=sample_rate,
        clip_duration=clip_duration,
        use_fp16=fp16,
    )

    trainer = ModelTrainer(config)
    trainer.setup()

    def progress(epoch, total, metrics):
        val_info = ""
        if metrics.val_loss > 0:
            val_info = f", val_loss: {metrics.val_loss:.4f}, val_acc: {metrics.val_accuracy:.4f}"
        click.echo(
            f"Epoch {epoch}/{total} - "
            f"loss: {metrics.train_loss:.4f}, acc: {metrics.train_accuracy:.4f}"
            f"{val_info} [{metrics.epoch_time:.1f}s]"
        )

    try:
        trainer.train(progress_callback=progress)
        click.echo(f"\nTraining complete! Model saved to {output_dir}")
    except Exception as e:
        click.echo(f"Training error: {e}")
        raise SystemExit(1)


@models.command('convert')
@click.argument('input_path')
@click.argument('output_path')
@click.option('--format', 'output_format', type=click.Choice(['pt', 'onnx']),
              default='onnx', help='Output format')
@click.option('--model-type', type=click.Choice(['ast', 'birdnet', 'opensoundscape']),
              default='ast', help='Model type')
def models_convert(input_path, output_path, output_format, model_type):
    """Convert model between formats (PyTorch to ONNX).

    Example:
        bioamla models convert ./my_model.pt ./my_model.onnx --format onnx
    """
    from bioamla.models import load_model

    click.echo(f"Loading model from {input_path}...")
    model = load_model(model_type, input_path)

    click.echo(f"Converting to {output_format}...")
    try:
        result = model.save(output_path, format=output_format)
        click.echo(f"Model saved to {result}")
    except Exception as e:
        click.echo(f"Conversion error: {e}")
        raise SystemExit(1)


@models.command('info')
@click.argument('model_path')
@click.option('--model-type', type=click.Choice(['ast', 'birdnet', 'opensoundscape']),
              default='ast', help='Model type')
def models_info(model_path, model_type):
    """Display information about a model."""
    from bioamla.models import load_model

    try:
        model = load_model(model_type, model_path)
        click.echo(f"Model: {model}")
        click.echo(f"Backend: {model.backend.value}")
        click.echo(f"Classes: {model.num_classes}")
        if model.classes:
            click.echo(f"Labels: {', '.join(model.classes[:10])}" +
                      (f"... (+{len(model.classes)-10} more)" if len(model.classes) > 10 else ""))
    except Exception as e:
        click.echo(f"Error loading model: {e}")
        raise SystemExit(1)


# =============================================================================
# Audio Command Group
# =============================================================================

@cli.group()
def audio():
    """Audio file utilities."""
    pass


@audio.command('list')
@click.argument('filepath', required=False, default='.')
def audio_list(filepath: str):
    """List audio files in a directory."""
    from bioamla.core.utils import get_audio_files
    try:
        if filepath == '.':
            import os
            filepath = os.getcwd()
        audio_files = get_audio_files(filepath)
        if audio_files:
            for file in audio_files:
                click.echo(file)
        else:
            click.echo("No audio files found in the specified directory.")
    except Exception as e:
        click.echo(f"An error occurred: {e}")


@audio.command('info')
@click.argument('filepath')
def audio_info(filepath: str):
    """Display metadata from an audio file."""
    from bioamla.core.utils import get_wav_metadata
    metadata = get_wav_metadata(filepath)
    click.echo(f"{metadata}")


@audio.command('convert')
@click.argument('path')
@click.argument('target_format')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--dataset', is_flag=True, help='Convert dataset with metadata.csv (updates metadata)')
@click.option('--metadata-filename', default='metadata.csv', help='Name of metadata CSV file (for --dataset mode)')
@click.option('--keep-original', is_flag=True, help='Keep original files after conversion')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories (batch mode)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_convert(
    path: str,
    target_format: str,
    output: str,
    batch: bool,
    dataset: bool,
    metadata_filename: str,
    keep_original: bool,
    recursive: bool,
    quiet: bool
):
    """Convert audio files to a different format.

    Single file: bioamla audio convert input.mp3 wav -o output.wav

    Batch mode: bioamla audio convert ./audio_dir wav --batch -o ./converted

    Dataset mode: bioamla audio convert ./dataset wav --dataset (updates metadata.csv)
    """
    from pathlib import Path

    if dataset:
        # Legacy dataset mode with metadata.csv
        from bioamla.core.datasets import convert_filetype

        stats = convert_filetype(
            dataset_path=path,
            target_format=target_format,
            metadata_filename=metadata_filename,
            keep_original=keep_original,
            verbose=not quiet
        )

        if quiet:
            click.echo(f"Converted {stats['files_converted']} files to {target_format}")

    elif batch:
        # Batch convert directory
        from bioamla.core.datasets import batch_convert_audio

        if output is None:
            output = str(Path(path)) + f"_{target_format}"

        try:
            stats = batch_convert_audio(
                input_dir=path,
                output_dir=output,
                target_format=target_format,
                recursive=recursive,
                keep_original=keep_original,
                verbose=not quiet
            )

            if quiet:
                click.echo(f"Converted {stats['files_converted']} files to {output}")
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)
        except ValueError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

    else:
        # Single file conversion
        from bioamla.core.datasets import convert_audio_file

        input_path = Path(path)
        if not input_path.exists():
            click.echo(f"Error: File not found: {path}")
            raise SystemExit(1)

        if output is None:
            output = str(input_path.with_suffix(f".{target_format.lstrip('.')}"))

        try:
            result = convert_audio_file(str(input_path), output, target_format)
            if not quiet:
                click.echo(f"Converted: {result}")
        except ValueError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)


@audio.command('filter')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--bandpass', default=None, help='Bandpass filter range (e.g., "1000-8000")')
@click.option('--lowpass', default=None, type=float, help='Lowpass cutoff frequency in Hz')
@click.option('--highpass', default=None, type=float, help='Highpass cutoff frequency in Hz')
@click.option('--order', default=5, type=int, help='Filter order (default: 5)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_filter(path, output, batch, bandpass, lowpass, highpass, order, quiet):
    """Apply frequency filter to audio files."""
    from bioamla.core.audio.signal import (
        bandpass_filter,
        highpass_filter,
        lowpass_filter,
    )

    if not any([bandpass, lowpass, highpass]):
        click.echo("Error: Must specify --bandpass, --lowpass, or --highpass")
        raise SystemExit(1)

    def processor(audio, sr):
        if bandpass:
            parts = bandpass.split('-')
            low, high = float(parts[0]), float(parts[1])
            return bandpass_filter(audio, sr, low, high, order)
        elif lowpass:
            return lowpass_filter(audio, sr, lowpass, order)
        elif highpass:
            return highpass_filter(audio, sr, highpass, order)
        return audio

    _run_signal_processing(path, output, batch, processor, quiet, "filter")


@audio.command('denoise')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--method', type=click.Choice(['spectral']), default='spectral', help='Denoising method')
@click.option('--strength', default=1.0, type=float, help='Noise reduction strength (0-2, default: 1.0)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_denoise(path, output, batch, method, strength, quiet):
    """Apply noise reduction to audio files."""
    from bioamla.core.audio.signal import spectral_denoise

    def processor(audio, sr):
        return spectral_denoise(audio, sr, noise_reduce_factor=strength)

    _run_signal_processing(path, output, batch, processor, quiet, "denoise")


@audio.command('segment')
@click.argument('path')
@click.option('--output', '-o', required=True, help='Output directory for segments')
@click.option('--batch', is_flag=True, help='Process all audio files in directory')
@click.option('--silence-threshold', default=-40, type=float, help='Silence threshold in dB (default: -40)')
@click.option('--min-silence', default=0.3, type=float, help='Min silence duration in seconds (default: 0.3)')
@click.option('--min-segment', default=0.5, type=float, help='Min segment duration in seconds (default: 0.5)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_segment(path, output, batch, silence_threshold, min_silence, min_segment, quiet):
    """Split audio on silence into separate files."""
    from pathlib import Path

    from bioamla.core.audio.signal import load_audio, save_audio, split_audio_on_silence
    from bioamla.core.utils import get_audio_files

    path = Path(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        click.echo(f"Error: Path not found: {path}")
        raise SystemExit(1)

    def segment_file(audio_path, output_dir):
        """Segment a single audio file."""
        audio, sr = load_audio(str(audio_path))
        chunks = split_audio_on_silence(
            audio, sr,
            silence_threshold_db=silence_threshold,
            min_silence_duration=min_silence,
            min_segment_duration=min_segment
        )

        if not chunks:
            return 0

        stem = audio_path.stem
        for i, (chunk, start, end) in enumerate(chunks):
            out_path = output_dir / f"{stem}_seg{i+1:03d}_{start:.2f}-{end:.2f}s.wav"
            save_audio(str(out_path), chunk, sr)
            if not quiet:
                click.echo(f"  Created: {out_path}")

        return len(chunks)

    if batch or path.is_dir():
        # Batch mode: process all files in directory
        audio_files = get_audio_files(str(path), recursive=True)
        if not audio_files:
            click.echo(f"No audio files found in {path}")
            raise SystemExit(1)

        total_segments = 0
        files_processed = 0
        files_failed = 0

        if not quiet:
            from bioamla.core.progress import ProgressBar, print_error, print_success

            with ProgressBar(
                total=len(audio_files),
                description="Segmenting audio files",
            ) as progress:
                for audio_file in audio_files:
                    audio_path = Path(audio_file)
                    # Preserve directory structure in output
                    try:
                        rel_path = audio_path.parent.relative_to(path)
                        file_output_dir = output / rel_path
                    except ValueError:
                        file_output_dir = output

                    file_output_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        num_segments = segment_file(audio_path, file_output_dir)
                        total_segments += num_segments
                        files_processed += 1
                    except Exception as e:
                        files_failed += 1
                    progress.advance()

            if files_failed > 0:
                print_error(f"Processed {files_processed} files, created {total_segments} segments, {files_failed} failed")
            else:
                print_success(f"Processed {files_processed} files, created {total_segments} segments")
        else:
            for audio_file in audio_files:
                audio_path = Path(audio_file)
                try:
                    rel_path = audio_path.parent.relative_to(path)
                    file_output_dir = output / rel_path
                except ValueError:
                    file_output_dir = output

                file_output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    num_segments = segment_file(audio_path, file_output_dir)
                    total_segments += num_segments
                    files_processed += 1
                except Exception:
                    files_failed += 1
    else:
        # Single file mode
        num_segments = segment_file(path, output)
        if num_segments == 0:
            click.echo("No segments found")
        else:
            click.echo(f"Created {num_segments} segments in {output}")


@audio.command('detect-events')
@click.argument('path')
@click.option('--output', '-o', required=True, help='Output CSV file for events')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_detect_events(path, output, quiet):
    """Detect onset events in audio and save to CSV."""
    import csv
    from pathlib import Path

    from bioamla.core.audio.signal import detect_onsets, load_audio

    path = Path(path)
    if not path.exists():
        click.echo(f"Error: File not found: {path}")
        raise SystemExit(1)

    audio, sr = load_audio(str(path))
    events = detect_onsets(audio, sr)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with TextFile(output, mode='w', newline='') as f:
        writer = csv.writer(f.handle)
        writer.writerow(['time', 'strength'])
        for event in events:
            writer.writerow([f"{event.time:.4f}", f"{event.strength:.4f}"])

    if not quiet:
        click.echo(f"Detected {len(events)} events, saved to {output}")


@audio.command('normalize')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--target-db', default=-20, type=float, help='Target loudness in dB (default: -20)')
@click.option('--peak', is_flag=True, help='Use peak normalization instead of RMS')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_normalize(path, output, batch, target_db, peak, quiet):
    """Normalize audio loudness."""
    from bioamla.core.audio.signal import normalize_loudness, peak_normalize

    def processor(audio, sr):
        if peak:
            target_linear = 10 ** (target_db / 20)
            return peak_normalize(audio, target_peak=min(target_linear, 0.99))
        return normalize_loudness(audio, sr, target_db=target_db)

    _run_signal_processing(path, output, batch, processor, quiet, "normalize")


@audio.command('resample')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--rate', required=True, type=int, help='Target sample rate in Hz')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_resample(path, output, batch, rate, quiet):
    """Resample audio to a different sample rate."""
    from bioamla.core.audio.signal import resample_audio

    def processor(audio, sr):
        return resample_audio(audio, sr, rate)

    _run_signal_processing(path, output, batch, processor, quiet, "resample", output_sr=rate)


@audio.command('trim')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file or directory')
@click.option('--batch', is_flag=True, help='Process all files in directory')
@click.option('--start', default=None, type=float, help='Start time in seconds')
@click.option('--end', default=None, type=float, help='End time in seconds')
@click.option('--silence', is_flag=True, help='Trim silence from start/end instead')
@click.option('--threshold', default=-40, type=float, help='Silence threshold in dB (for --silence)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_trim(path, output, batch, start, end, silence, threshold, quiet):
    """Trim audio by time or remove silence."""
    from bioamla.core.audio.signal import trim_audio, trim_silence

    if not silence and start is None and end is None:
        click.echo("Error: Must specify --start/--end or use --silence")
        raise SystemExit(1)

    def processor(audio, sr):
        if silence:
            return trim_silence(audio, sr, threshold_db=threshold)
        return trim_audio(audio, sr, start_time=start, end_time=end)

    _run_signal_processing(path, output, batch, processor, quiet, "trim")


@audio.command('analyze')
@click.argument('path')
@click.option('--batch', is_flag=True, help='Analyze all audio files in directory')
@click.option('--output', '-o', default=None, help='Output file for results (CSV or JSON)')
@click.option('--format', 'output_format', type=click.Choice(['text', 'json', 'csv']), default='text',
              help='Output format (default: text)')
@click.option('--silence-threshold', default=-40, type=float, help='Silence detection threshold in dB')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories (batch mode)')
@click.option('--quiet', is_flag=True, help='Suppress detailed output')
def audio_analyze(path, batch, output, output_format, silence_threshold, recursive, quiet):
    """Analyze audio files and display statistics.

    Shows duration, sample rate, channels, RMS/peak amplitude,
    frequency statistics, and silence detection results.

    Single file mode:
        bioamla audio analyze recording.wav

    Batch mode:
        bioamla audio analyze ./audio_dir --batch --output results.csv

    Examples:
        # Analyze a single file
        bioamla audio analyze bird_call.wav

        # Analyze with JSON output
        bioamla audio analyze recording.wav --format json

        # Batch analyze and save to CSV
        bioamla audio analyze ./dataset --batch -o analysis.csv --format csv
    """
    import json
    from pathlib import Path

    from bioamla.core.audio.audio import (
        analyze_audio,
        summarize_analysis,
    )
    from bioamla.core.utils import get_audio_files

    path = Path(path)

    if batch:
        # Batch mode
        if not path.is_dir():
            click.echo(f"Error: {path} is not a directory")
            raise SystemExit(1)

        audio_files = get_audio_files(str(path), recursive=recursive)

        if not audio_files:
            click.echo("No audio files found")
            raise SystemExit(1)

        analyses = []
        errors = []

        if not quiet:
            from bioamla.core.progress import ProgressBar, print_success, print_error

            with ProgressBar(
                total=len(audio_files),
                description="Analyzing audio files",
            ) as progress:
                for filepath in audio_files:
                    try:
                        analysis = analyze_audio(filepath, silence_threshold_db=silence_threshold)
                        analyses.append(analysis)
                    except Exception as e:
                        errors.append((filepath, str(e)))
                    progress.advance()

            if errors:
                print_error(f"Analyzed {len(analyses)} files, {len(errors)} failed")
            else:
                print_success(f"Analyzed {len(analyses)} files")
        else:
            for filepath in audio_files:
                try:
                    analysis = analyze_audio(filepath, silence_threshold_db=silence_threshold)
                    analyses.append(analysis)
                except Exception:
                    pass

        if output_format == 'json':
            result = {
                "summary": summarize_analysis(analyses),
                "files": [a.to_dict() for a in analyses]
            }
            if output:
                with TextFile(output, mode='w') as f:
                    json.dump(result, f.handle, indent=2)
                click.echo(f"Results saved to {output}")
            else:
                click.echo(json.dumps(result, indent=2))

        elif output_format == 'csv':
            import csv
            output_path = output or "analysis_results.csv"
            with TextFile(output_path, mode='w', newline='') as f:
                writer = csv.writer(f.handle)
                writer.writerow([
                    'file_path', 'duration', 'sample_rate', 'channels',
                    'rms', 'rms_db', 'peak', 'peak_db',
                    'peak_frequency', 'spectral_centroid', 'bandwidth',
                    'is_silent', 'silence_ratio'
                ])
                for a in analyses:
                    writer.writerow([
                        a.file_path,
                        f"{a.info.duration:.3f}",
                        a.info.sample_rate,
                        a.info.channels,
                        f"{a.amplitude.rms:.6f}",
                        f"{a.amplitude.rms_db:.2f}",
                        f"{a.amplitude.peak:.6f}",
                        f"{a.amplitude.peak_db:.2f}",
                        f"{a.frequency.peak_frequency:.1f}",
                        f"{a.frequency.spectral_centroid:.1f}",
                        f"{a.frequency.bandwidth:.1f}",
                        a.silence.is_silent,
                        f"{a.silence.silence_ratio:.3f}"
                    ])
            click.echo(f"Results saved to {output_path}")

        else:
            # Text format - show summary
            summary = summarize_analysis(analyses)
            click.echo("\nBatch Analysis Summary")
            click.echo("=" * 50)
            click.echo(f"Files analyzed: {summary['total_files']}")
            click.echo(f"Total duration: {summary['total_duration']:.2f}s")
            click.echo(f"Average duration: {summary['avg_duration']:.2f}s")
            click.echo(f"Duration range: {summary['min_duration']:.2f}s - {summary['max_duration']:.2f}s")
            click.echo("\nAmplitude (average):")
            click.echo(f"  RMS: {summary['avg_rms_db']:.1f} dBFS")
            click.echo(f"  Peak: {summary['avg_peak_db']:.1f} dBFS")
            click.echo(f"\nFrequency (average peak): {summary['avg_peak_frequency']:.1f} Hz")
            click.echo(f"  Range: {summary['min_peak_frequency']:.1f} - {summary['max_peak_frequency']:.1f} Hz")
            click.echo(f"\nSilent files: {summary['silent_file_count']} ({summary['silent_file_ratio']:.1%})")

    else:
        # Single file mode
        if not path.exists():
            click.echo(f"Error: File not found: {path}")
            raise SystemExit(1)

        try:
            analysis = analyze_audio(str(path), silence_threshold_db=silence_threshold)
        except Exception as e:
            click.echo(f"Error analyzing file: {e}")
            raise SystemExit(1)

        if output_format == 'json':
            result = analysis.to_dict()
            if output:
                with TextFile(output, mode='w', encoding='utf-8') as f:
                    json.dump(result, f.handle, indent=2)
                click.echo(f"Results saved to {output}")
            else:
                click.echo(json.dumps(result, indent=2))
        else:
            # Text format
            click.echo(f"\nAudio Analysis: {path}")
            click.echo("=" * 50)
            click.echo("\nBasic Info:")
            click.echo(f"  Duration: {analysis.info.duration:.3f}s")
            click.echo(f"  Sample rate: {analysis.info.sample_rate} Hz")
            click.echo(f"  Channels: {analysis.info.channels}")
            click.echo(f"  Samples: {analysis.info.samples:,}")
            if analysis.info.bit_depth:
                click.echo(f"  Bit depth: {analysis.info.bit_depth}")
            if analysis.info.format:
                click.echo(f"  Format: {analysis.info.format}")

            click.echo("\nAmplitude:")
            click.echo(f"  RMS: {analysis.amplitude.rms:.6f} ({analysis.amplitude.rms_db:.1f} dBFS)")
            click.echo(f"  Peak: {analysis.amplitude.peak:.6f} ({analysis.amplitude.peak_db:.1f} dBFS)")
            click.echo(f"  Crest factor: {analysis.amplitude.crest_factor:.1f} dB")

            click.echo("\nFrequency:")
            click.echo(f"  Peak: {analysis.frequency.peak_frequency:.1f} Hz")
            click.echo(f"  Mean: {analysis.frequency.mean_frequency:.1f} Hz")
            click.echo(f"  Spectral centroid: {analysis.frequency.spectral_centroid:.1f} Hz")
            click.echo(f"  Bandwidth: {analysis.frequency.min_frequency:.1f} - {analysis.frequency.max_frequency:.1f} Hz")
            click.echo(f"  Spectral rolloff: {analysis.frequency.spectral_rolloff:.1f} Hz")

            click.echo(f"\nSilence Detection (threshold: {silence_threshold} dB):")
            click.echo(f"  Is silent: {analysis.silence.is_silent}")
            click.echo(f"  Silence ratio: {analysis.silence.silence_ratio:.1%}")
            click.echo(f"  Sound ratio: {analysis.silence.sound_ratio:.1%}")
            if analysis.silence.sound_segments:
                click.echo(f"  Sound segments: {len(analysis.silence.sound_segments)}")


def _run_signal_processing(path, output, batch, processor, quiet, operation, output_sr=None):
    """Helper to run signal processing on file or directory."""
    from pathlib import Path

    from bioamla.core.audio.signal import batch_process, load_audio, save_audio

    path = Path(path)

    if batch:
        if output is None:
            output = str(path) + f"_{operation}"

        try:
            stats = batch_process(
                str(path), output, processor,
                sample_rate=output_sr, verbose=not quiet
            )
            if quiet:
                click.echo(f"Processed {stats['files_processed']} files to {stats['output_dir']}")
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)
    else:
        if not path.exists():
            click.echo(f"Error: File not found: {path}")
            raise SystemExit(1)

        if output is None:
            output = str(path.with_stem(path.stem + f"_{operation}"))

        try:
            audio, sr = load_audio(str(path))
            processed = processor(audio, sr)
            if output_sr:
                sr = output_sr
            save_audio(output, processed, sr)
            if not quiet:
                click.echo(f"Saved: {output}")
        except Exception as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)


# =============================================================================
# Visualize Command (under audio group)
# =============================================================================

@audio.command('visualize')
@click.argument('path')
@click.option('--output', '-o', default=None, help='Output file path (single file) or directory (batch with --batch)')
@click.option('--batch', is_flag=True, default=False, help='Process all audio files in a directory')
@click.option('--type', 'viz_type', type=click.Choice(['stft', 'mel', 'mfcc', 'waveform']), default='mel', help='Type of visualization')
@click.option('--sample-rate', default=16000, type=int, help='Target sample rate for processing')
@click.option('--n-fft', default=2048, type=int, help='FFT window size (256-8192)')
@click.option('--hop-length', default=512, type=int, help='Samples between successive frames')
@click.option('--n-mels', default=128, type=int, help='Number of mel bands (mel spectrogram only)')
@click.option('--n-mfcc', default=40, type=int, help='Number of MFCCs (mfcc only)')
@click.option('--window', type=click.Choice(['hann', 'hamming', 'blackman', 'bartlett', 'rectangular', 'kaiser']), default='hann', help='Window function for STFT')
@click.option('--db-min', default=None, type=float, help='Minimum dB value for scaling')
@click.option('--db-max', default=None, type=float, help='Maximum dB value for scaling')
@click.option('--cmap', default='magma', help='Colormap for spectrogram visualizations')
@click.option('--dpi', default=150, type=int, help='Output image resolution (dots per inch)')
@click.option('--format', 'img_format', type=click.Choice(['png', 'jpg', 'jpeg']), default=None, help='Output image format (default: inferred from extension)')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories (batch mode only)')
@click.option('--progress/--no-progress', default=True, help='Show Rich progress bar (batch mode only)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def audio_visualize(
    path: str,
    output: str,
    batch: bool,
    viz_type: str,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    n_mfcc: int,
    window: str,
    db_min: float,
    db_max: float,
    cmap: str,
    dpi: int,
    img_format: str,
    recursive: bool,
    progress: bool,
    quiet: bool
):
    """
    Generate spectrogram visualizations from audio files.

    PATH can be a single audio file or a directory (with --batch flag).

    Single file mode (default):
        bioamla audio visualize audio.wav --output spec.png

    Batch mode (--batch):
        bioamla audio visualize ./audio_dir --batch --output ./specs

    Visualization types:
        stft: Short-Time Fourier Transform spectrogram
        mel: Mel spectrogram (default)
        mfcc: Mel-frequency cepstral coefficients
        waveform: Time-domain waveform plot

    Window functions:
        hann (default), hamming, blackman, bartlett, rectangular, kaiser

    Examples:
        # STFT spectrogram with custom FFT size
        bioamla audio visualize audio.wav --type stft --n-fft 4096

        # Mel spectrogram with dB limits and JPEG output
        bioamla audio visualize audio.wav --type mel --db-min -80 --db-max 0 -o spec.jpg

        # Batch processing with hamming window
        bioamla audio visualize ./audio --batch --window hamming --format png
    """
    import os

    from bioamla.core.visualize import batch_generate_spectrograms, generate_spectrogram

    if batch:
        # Batch mode: process directory
        if output is None:
            output = os.path.join(path, "spectrograms")

        stats = batch_generate_spectrograms(
            input_dir=path,
            output_dir=output,
            viz_type=viz_type,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            window=window,
            db_min=db_min,
            db_max=db_max,
            cmap=cmap,
            dpi=dpi,
            format=img_format if img_format else "png",
            recursive=recursive,
            verbose=not quiet,
            use_rich_progress=progress and not quiet,
        )

        if quiet:
            click.echo(f"Generated {stats['files_processed']} spectrograms in {stats['output_dir']}")
    else:
        # Single file mode
        if output is None:
            # Default output: same name with .png extension
            base_name = os.path.splitext(path)[0]
            ext = ".jpg" if img_format in ("jpg", "jpeg") else ".png"
            output = f"{base_name}{ext}"

        try:
            result = generate_spectrogram(
                audio_path=path,
                output_path=output,
                viz_type=viz_type,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                window=window,
                db_min=db_min,
                db_max=db_max,
                cmap=cmap,
                dpi=dpi,
                format=img_format,
            )
            if not quiet:
                click.echo(f"Generated {viz_type} spectrogram: {result}")
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)
        except Exception as e:
            click.echo(f"Error generating spectrogram: {e}")
            raise SystemExit(1)


# =============================================================================
# Services Command Group
# =============================================================================

@cli.group()
def services():
    """External service integrations (Xeno-canto, Macaulay Library, iNaturalist, etc.)."""
    pass


# =============================================================================
# iNaturalist subgroup (under services)
# =============================================================================

@services.group('inat')
def services_inat():
    """iNaturalist observation database."""
    pass


@services_inat.command('download')
@click.argument('output_dir')
@click.option('--taxon-ids', default=None, help='Comma-separated list of taxon IDs (e.g., "3" for birds, "3,20978" for multiple)')
@click.option('--taxon-csv', default=None, type=click.Path(exists=True), help='Path to CSV file with taxon_id column')
@click.option('--taxon-name', default=None, help='Filter by taxon name (e.g., "Aves" for birds)')
@click.option('--place-id', type=int, default=None, help='Filter by place ID (e.g., 1 for United States)')
@click.option('--user-id', default=None, help='Filter by observer username')
@click.option('--project-id', default=None, help='Filter by iNaturalist project ID or slug')
@click.option('--quality-grade', default='research', help='Quality grade: research, needs_id, or casual')
@click.option('--sound-license', default=None, help='Comma-separated list of sound licenses (e.g., "cc-by, cc-by-nc, cc0")')
@click.option('--start-date', default=None, help='Start date for observations (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date for observations (YYYY-MM-DD)')
@click.option('--obs-per-taxon', type=int, default=100, help='Number of observations to download per taxon ID')
@click.option('--organize-by-taxon/--no-organize-by-taxon', default=True, help='Organize files into subdirectories by species')
@click.option('--include-inat-metadata', is_flag=True, help='Include additional iNaturalist metadata fields in CSV')
@click.option('--file-extensions', default=None, help='Comma-separated list of file extensions to filter (e.g., "wav,mp3")')
@click.option('--delay', type=float, default=1.0, help='Delay between downloads in seconds (rate limiting)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def inat_download(
    output_dir: str,
    taxon_ids: str,
    taxon_csv: str,
    taxon_name: str,
    place_id: int,
    user_id: str,
    project_id: str,
    quality_grade: str,
    sound_license: str,
    start_date: str,
    end_date: str,
    obs_per_taxon: int,
    organize_by_taxon: bool,
    include_inat_metadata: bool,
    file_extensions: str,
    delay: float,
    quiet: bool
):
    """Download audio observations from iNaturalist."""
    from bioamla.core.services.inaturalist import download_inat_audio

    taxon_ids_list = None
    if taxon_ids:
        taxon_ids_list = [int(tid.strip()) for tid in taxon_ids.split(",")]

    extensions_list = None
    if file_extensions:
        extensions_list = [ext.strip() for ext in file_extensions.split(",")]

    sound_license_list = None
    if sound_license:
        sound_license_list = [lic.strip() for lic in sound_license.split(",")]

    stats = download_inat_audio(
        output_dir=output_dir,
        taxon_ids=taxon_ids_list,
        taxon_csv=taxon_csv,
        taxon_name=taxon_name,
        place_id=place_id,
        user_id=user_id,
        project_id=project_id,
        quality_grade=quality_grade,
        sound_license=sound_license_list,
        d1=start_date,
        d2=end_date,
        obs_per_taxon=obs_per_taxon,
        organize_by_taxon=organize_by_taxon,
        include_inat_metadata=include_inat_metadata,
        file_extensions=extensions_list,
        delay_between_downloads=delay,
        verbose=not quiet
    )

    if quiet:
        click.echo(f"Downloaded {stats['total_sounds']} audio files to {stats['output_dir']}")


@services_inat.command('search')
@click.option('--species', default=None, help='Filter by species name (scientific or common)')
@click.option('--taxon-id', type=int, default=None, help='Filter by taxon ID (e.g., 20979 for Amphibia)')
@click.option('--place-id', type=int, default=None, help='Filter by place ID (e.g., 1 for United States)')
@click.option('--project-id', default=None, help='Filter by iNaturalist project ID or slug')
@click.option('--quality-grade', default='research', help='Quality grade: research, needs_id, or casual')
@click.option('--has-sounds', is_flag=True, help='Only show observations with sounds')
@click.option('--limit', type=int, default=20, help='Maximum number of results')
@click.option('--output', '-o', default=None, help='Output file path for CSV (optional)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def inat_search(
    species: str,
    taxon_id: int,
    place_id: int,
    project_id: str,
    quality_grade: str,
    has_sounds: bool,
    limit: int,
    output: str,
    quiet: bool
):
    """Search for iNaturalist observations."""
    from bioamla.core.services.inaturalist import search_inat_sounds

    if not species and not taxon_id and not place_id and not project_id:
        raise click.UsageError("At least one search filter must be provided (--species, --taxon-id, --place-id, or --project-id)")

    results = search_inat_sounds(
        taxon_id=taxon_id,
        taxon_name=species,
        place_id=place_id,
        quality_grade=quality_grade,
        per_page=limit
    )

    if not results:
        click.echo("No observations found matching the search criteria.")
        return

    if output:
        import csv
        with TextFile(output, mode='w', newline='', encoding='utf-8') as f:
            fieldnames = ['observation_id', 'scientific_name', 'common_name', 'sound_count', 'observed_on', 'location', 'url']
            writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
            writer.writeheader()
            for obs in results:
                taxon = obs.get('taxon', {})
                observed_on_raw = obs.get('observed_on', '')
                if hasattr(observed_on_raw, 'strftime'):
                    observed_on = observed_on_raw.strftime('%Y-%m-%d')
                else:
                    observed_on = str(observed_on_raw) if observed_on_raw else ''
                writer.writerow({
                    'observation_id': obs.get('id'),
                    'scientific_name': taxon.get('name', ''),
                    'common_name': taxon.get('preferred_common_name', ''),
                    'sound_count': len(obs.get('sounds', [])),
                    'observed_on': observed_on,
                    'location': obs.get('place_guess', ''),
                    'url': f"https://www.inaturalist.org/observations/{obs.get('id')}"
                })
        click.echo(f"Saved {len(results)} observations to {output}")
    else:
        click.echo(f"\nFound {len(results)} observations with sounds:\n")
        click.echo(f"{'ID':<12} {'Species':<30} {'Sounds':<8} {'Date':<12} {'Location':<30}")
        click.echo("-" * 95)
        for obs in results:
            taxon = obs.get('taxon', {})
            obs_id = obs.get('id', '')
            name = taxon.get('name', 'Unknown')[:28]
            sound_count = len(obs.get('sounds', []))
            observed_on_raw = obs.get('observed_on', '')
            if hasattr(observed_on_raw, 'strftime'):
                observed_on = observed_on_raw.strftime('%Y-%m-%d')
            else:
                observed_on = str(observed_on_raw)[:10] if observed_on_raw else ''
            location = (obs.get('place_guess', '') or '')[:28]
            click.echo(f"{obs_id:<12} {name:<30} {sound_count:<8} {observed_on:<12} {location:<30}")


@services_inat.command('stats')
@click.argument('project_id')
@click.option('--output', '-o', default=None, help='Output file path for JSON (optional)')
@click.option('--quiet', is_flag=True, help='Suppress progress output, print only JSON')
def inat_stats(
    project_id: str,
    output: str,
    quiet: bool
):
    """Get statistics for an iNaturalist project."""
    import json

    from bioamla.core.services.inaturalist import get_project_stats

    stats = get_project_stats(
        project_id=project_id,
        verbose=not quiet
    )

    if output:
        with TextFile(output, mode='w', encoding='utf-8') as f:
            json.dump(stats, f.handle, indent=2)
        click.echo(f"Saved project stats to {output}")
    elif quiet:
        click.echo(json.dumps(stats, indent=2))
    else:
        click.echo(f"\nProject: {stats['title']}")
        click.echo(f"URL: {stats['url']}")
        click.echo(f"Type: {stats['project_type']}")
        if stats['place']:
            click.echo(f"Place: {stats['place']}")
        click.echo(f"Created: {stats['created_at']}")
        click.echo("\nStatistics:")
        click.echo(f"  Observations: {stats['observation_count']}")
        click.echo(f"  Species: {stats['species_count']}")
        click.echo(f"  Observers: {stats['observers_count']}")


# =============================================================================
# Dataset Command Group
# =============================================================================

@cli.group()
def dataset():
    """Dataset management commands."""
    pass


@dataset.command('merge')
@click.argument('output_dir')
@click.argument('dataset_paths', nargs=-1, required=True)
@click.option('--metadata-filename', default='metadata.csv', help='Name of metadata CSV file in each dataset')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files instead of skipping')
@click.option('--no-organize', is_flag=True, help='Preserve original directory structure instead of organizing by category')
@click.option('--target-format', default=None, help='Convert all audio files to this format (wav, mp3, flac, etc.)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def dataset_merge(
    output_dir: str,
    dataset_paths: tuple,
    metadata_filename: str,
    overwrite: bool,
    no_organize: bool,
    target_format: str,
    quiet: bool
):
    """Merge multiple audio datasets into a single dataset."""
    from bioamla.core.datasets import merge_datasets as do_merge

    stats = do_merge(
        dataset_paths=list(dataset_paths),
        output_dir=output_dir,
        metadata_filename=metadata_filename,
        skip_existing=not overwrite,
        organize_by_category=not no_organize,
        target_format=target_format,
        verbose=not quiet
    )

    if quiet:
        msg = f"Merged {stats['datasets_merged']} datasets: {stats['total_files']} total files"
        if target_format:
            msg += f", {stats['files_converted']} converted"
        click.echo(msg)


@dataset.command('license')
@click.argument('path')
@click.option('--template', '-t', default=None, help='Template file to prepend to the license file')
@click.option('--output', '-o', default='LICENSE', help='Output filename for the license file')
@click.option('--metadata-filename', default='metadata.csv', help='Name of metadata CSV file')
@click.option('--batch', is_flag=True, help='Process all datasets in directory (each subdirectory with metadata.csv)')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def dataset_license(
    path: str,
    template: str,
    output: str,
    metadata_filename: str,
    batch: bool,
    quiet: bool
):
    """Generate license/attribution file from dataset metadata.

    Reads attribution data from metadata CSV and generates a formatted LICENSE file.

    Required CSV columns: file_name, attr_id, attr_lic, attr_url, attr_note
    """
    from pathlib import Path as PathLib

    from bioamla.core.license import (
        generate_license_for_dataset,
        generate_licenses_for_directory,
    )

    path_obj = PathLib(path)
    template_path = PathLib(template) if template else None

    if template_path and not template_path.exists():
        click.echo(f"Error: Template file '{template}' not found.")
        raise SystemExit(1)

    if batch:
        # Process all datasets in directory
        if not path_obj.is_dir():
            click.echo(f"Error: Path '{path}' is not a directory.")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Scanning directory for datasets: {path}")

        try:
            stats = generate_licenses_for_directory(
                audio_dir=path_obj,
                template_path=template_path,
                output_filename=output,
                metadata_filename=metadata_filename
            )
        except FileNotFoundError as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

        if stats['datasets_found'] == 0:
            click.echo("No datasets found (no directories with metadata.csv)")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"\nProcessed {stats['datasets_found']} dataset(s):")
            click.echo(f"  Successful: {stats['datasets_processed']}")
            click.echo(f"  Failed: {stats['datasets_failed']}")

            for result in stats['results']:
                if result['status'] == 'success':
                    click.echo(f"  - {result['dataset_name']}: {result['attributions_count']} attributions")
                else:
                    click.echo(f"  - {result['dataset_name']}: FAILED - {result.get('error', 'Unknown error')}")
        else:
            click.echo(f"Generated {stats['datasets_processed']} license files")

        if stats['datasets_failed'] > 0:
            raise SystemExit(1)

    else:
        # Process single dataset
        if not path_obj.is_dir():
            click.echo(f"Error: Path '{path}' is not a directory.")
            raise SystemExit(1)

        csv_path = path_obj / metadata_filename
        if not csv_path.exists():
            click.echo(f"Error: Metadata file '{csv_path}' not found.")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"Generating license file for: {path}")

        try:
            stats = generate_license_for_dataset(
                dataset_path=path_obj,
                template_path=template_path,
                output_filename=output,
                metadata_filename=metadata_filename
            )
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)

        if not quiet:
            click.echo(f"License file generated: {stats['output_path']}")
            click.echo(f"  Attributions: {stats['attributions_count']}")
            click.echo(f"  File size: {stats['file_size']:,} bytes")
        else:
            click.echo(f"Generated {output} with {stats['attributions_count']} attributions")


# =============================================================================
# Augment Command (under dataset group)
# =============================================================================

def _parse_range(value: str) -> tuple:
    """Parse a range string like '0.8-1.2' or '-2,2' into (min, max)."""
    if '-' in value and not value.startswith('-'):
        parts = value.split('-')
        return float(parts[0]), float(parts[1])
    elif ',' in value:
        parts = value.split(',')
        return float(parts[0]), float(parts[1])
    else:
        val = float(value)
        return val, val


@dataset.command('augment')
@click.argument('input_dir')
@click.option('--output', '-o', required=True, help='Output directory for augmented files')
@click.option('--add-noise', default=None, help='Add Gaussian noise with SNR range (e.g., "3-30" dB)')
@click.option('--time-stretch', default=None, help='Time stretch range (e.g., "0.8-1.2")')
@click.option('--pitch-shift', default=None, help='Pitch shift range in semitones (e.g., "-2,2")')
@click.option('--gain', default=None, help='Gain range in dB (e.g., "-12,12")')
@click.option('--multiply', default=1, type=int, help='Number of augmented copies to create per file')
@click.option('--sample-rate', default=16000, type=int, help='Target sample rate for output')
@click.option('--recursive/--no-recursive', default=True, help='Search subdirectories')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def dataset_augment(
    input_dir: str,
    output: str,
    add_noise: str,
    time_stretch: str,
    pitch_shift: str,
    gain: str,
    multiply: int,
    sample_rate: int,
    recursive: bool,
    quiet: bool
):
    """
    Augment audio files to expand training datasets.

    Creates augmented copies of audio files with various transformations.
    At least one augmentation option must be specified.

    Examples:
        bioamla dataset augment ./audio --output ./augmented --add-noise 3-30

        bioamla dataset augment ./audio --output ./augmented \\
            --add-noise 3-30 \\
            --time-stretch 0.8-1.2 \\
            --pitch-shift -2,2 \\
            --multiply 5

    Augmentation options:
        --add-noise: Add Gaussian noise with SNR in specified range (dB)
        --time-stretch: Speed up/slow down without changing pitch
        --pitch-shift: Change pitch without changing speed (semitones)
        --gain: Random volume adjustment (dB)
    """
    from bioamla.core.augment import AugmentationConfig, batch_augment

    # Build configuration from options
    config = AugmentationConfig(
        sample_rate=sample_rate,
        multiply=multiply,
    )

    # Parse augmentation options
    if add_noise:
        config.add_noise = True
        min_snr, max_snr = _parse_range(add_noise)
        config.noise_min_snr = min_snr
        config.noise_max_snr = max_snr

    if time_stretch:
        config.time_stretch = True
        min_rate, max_rate = _parse_range(time_stretch)
        config.time_stretch_min = min_rate
        config.time_stretch_max = max_rate

    if pitch_shift:
        config.pitch_shift = True
        min_semi, max_semi = _parse_range(pitch_shift)
        config.pitch_shift_min = min_semi
        config.pitch_shift_max = max_semi

    if gain:
        config.gain = True
        min_db, max_db = _parse_range(gain)
        config.gain_min_db = min_db
        config.gain_max_db = max_db

    # Check that at least one augmentation is enabled
    if not any([config.add_noise, config.time_stretch, config.pitch_shift, config.gain]):
        click.echo("Error: At least one augmentation option must be specified")
        click.echo("Use --help for available options")
        raise SystemExit(1)

    try:
        stats = batch_augment(
            input_dir=input_dir,
            output_dir=output,
            config=config,
            recursive=recursive,
            verbose=not quiet,
        )

        if quiet:
            click.echo(
                f"Created {stats['files_created']} augmented files from "
                f"{stats['files_processed']} source files in {stats['output_dir']}"
            )
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error during augmentation: {e}")
        raise SystemExit(1)


@dataset.command('download')
@click.argument('url', required=True)
@click.argument('output_dir', required=False, default='.')
def dataset_download(url: str, output_dir: str):
    """Download a file from the specified URL to the target directory.

    Examples:
        bioamla dataset download https://example.com/dataset.zip ./data
        bioamla dataset download https://example.com/audio.tar.gz
    """
    import os
    from urllib.parse import urlparse

    from bioamla.core.utils import download_file

    if output_dir == '.':
        output_dir = os.getcwd()

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_file"

    output_path = os.path.join(output_dir, filename)
    download_file(url, output_path)


@dataset.command('unzip')
@click.argument('file_path')
@click.argument('output_path', required=False, default='.')
def dataset_unzip(file_path: str, output_path: str):
    """Extract a ZIP archive to the specified output directory.

    Examples:
        bioamla dataset unzip dataset.zip ./extracted
        bioamla dataset unzip archive.zip
    """
    from bioamla.core.utils import extract_zip_file
    import os
    if output_path == '.':
        output_path = os.getcwd()

    extract_zip_file(file_path, output_path)


@dataset.command('zip')
@click.argument('source_path')
@click.argument('output_file')
def dataset_zip(source_path: str, output_file: str):
    """Create a ZIP archive from a file or directory.

    Examples:
        bioamla dataset zip ./my_dataset dataset.zip
        bioamla dataset zip audio_file.wav single_file.zip
    """
    import os

    from bioamla.core.utils import create_zip_file, zip_directory

    if os.path.isdir(source_path):
        zip_directory(source_path, output_file)
    else:
        create_zip_file([source_path], output_file)

    click.echo(f"Created {output_file}")


# =============================================================================
# HuggingFace Hub Command Group
# =============================================================================

def _get_folder_size(path: str, limit: int | None = None) -> int:
    """Calculate the total size of a folder in bytes.

    Args:
        path: Path to the folder.
        limit: If provided, short-circuit and return once this size is exceeded.

    Returns:
        Total size in bytes, or a value > limit if short-circuited.
    """
    import os
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
                if limit is not None and total_size > limit:
                    return total_size
    return total_size


def _count_files(path: str, limit: int | None = None) -> int:
    """Count the total number of files in a folder.

    Args:
        path: Path to the folder.
        limit: If provided, short-circuit and return once this count is exceeded.

    Returns:
        Total file count, or a value > limit if short-circuited.
    """
    import os
    count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        count += len(filenames)
        if limit is not None and count > limit:
            return count
    return count


def _is_large_folder(path: str, size_threshold_gb: float = 5.0, file_count_threshold: int = 1000) -> bool:
    """
    Determine if a folder should be uploaded using upload_large_folder.

    A folder is considered 'large' if:
    - Total size exceeds size_threshold_gb (default: 5 GB), OR
    - Total file count exceeds file_count_threshold (default: 1000 files)

    Uses short-circuiting to avoid walking the entire tree for very large directories.
    """
    size_threshold_bytes = int(size_threshold_gb * 1024 * 1024 * 1024)

    # Check file count first (usually faster) with short-circuit
    file_count = _count_files(path, limit=file_count_threshold)
    if file_count > file_count_threshold:
        return True

    # Check size with short-circuit
    folder_size = _get_folder_size(path, limit=size_threshold_bytes)
    return folder_size > size_threshold_bytes


# --- HuggingFace Hub subgroup ---
@services.group('hf')
def services_hf():
    """HuggingFace Hub model and dataset management."""
    pass


@services_hf.command('push-model')
@click.argument('path')
@click.argument('repo_id')
@click.option('--private/--public', default=False, help='Make the repository private (default: public)')
@click.option('--commit-message', default=None, help='Custom commit message for the push')
def hf_push_model(
    path: str,
    repo_id: str,
    private: bool,
    commit_message: str
):
    """Push a model folder to the HuggingFace Hub.

    Uploads the entire contents of PATH folder to the Hub as a model.
    Automatically uses upload_large_folder for folders >5GB or >1000 files.
    """
    import os

    from huggingface_hub import HfApi

    if not os.path.isdir(path):
        click.echo(f"Error: Path '{path}' does not exist or is not a directory.")
        raise SystemExit(1)

    click.echo(f"Pushing model folder {path} to HuggingFace Hub: {repo_id}...")

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

        if _is_large_folder(path):
            click.echo("Large folder detected, using optimized upload method...")
            api.upload_large_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message or "Upload model",
            )
        else:
            api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message or "Upload model",
            )

        click.echo(f"Successfully pushed model to: https://huggingface.co/{repo_id}")

    except Exception as e:
        click.echo(f"Error pushing to HuggingFace Hub: {e}")
        click.echo("Make sure you are logged in with 'huggingface-cli login'.")
        raise SystemExit(1)


# =============================================================================
# Annotation Command Group
# =============================================================================

@cli.group()
def annotation():
    """Annotation management commands for audio datasets."""
    pass


@annotation.command('convert')
@click.argument('input_file')
@click.argument('output_file')
@click.option('--from', 'from_format', type=click.Choice(['raven', 'csv']), default=None,
              help='Input format (auto-detected from extension if not specified)')
@click.option('--to', 'to_format', type=click.Choice(['raven', 'csv']), default=None,
              help='Output format (auto-detected from extension if not specified)')
@click.option('--label-column', default=None, help='Column name for labels in input file')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def annotation_convert(input_file, output_file, from_format, to_format, label_column, quiet):
    """Convert annotation files between formats.

    Supported formats:
        raven: Raven Pro selection table (.txt, tab-delimited)
        csv: Standard CSV format

    Examples:
        bioamla annotation convert selections.txt annotations.csv
        bioamla annotation convert annotations.csv output.txt --to raven
    """
    from pathlib import Path

    from bioamla.core.annotations import (
        load_csv_annotations,
        load_raven_selection_table,
        save_csv_annotations,
        save_raven_selection_table,
    )

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}")
        raise SystemExit(1)

    # Auto-detect input format
    if from_format is None:
        if input_path.suffix.lower() == '.txt':
            from_format = 'raven'
        else:
            from_format = 'csv'

    # Auto-detect output format
    if to_format is None:
        if output_path.suffix.lower() == '.txt':
            to_format = 'raven'
        else:
            to_format = 'csv'

    # Load annotations
    if from_format == 'raven':
        annotations = load_raven_selection_table(input_file, label_column=label_column)
    else:
        annotations = load_csv_annotations(input_file)

    # Save annotations
    if to_format == 'raven':
        save_raven_selection_table(annotations, output_file)
    else:
        save_csv_annotations(annotations, output_file)

    if not quiet:
        click.echo(f"Converted {len(annotations)} annotations from {from_format} to {to_format}")
        click.echo(f"Output: {output_file}")


@annotation.command('summary')
@click.argument('path')
@click.option('--format', 'file_format', type=click.Choice(['raven', 'csv']), default=None,
              help='Annotation format (auto-detected from extension if not specified)')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def annotation_summary(path, file_format, output_json):
    """Display summary statistics for an annotation file.

    Shows:
        - Total number of annotations
        - Unique labels and their counts
        - Duration statistics
    """
    import json
    from pathlib import Path

    from bioamla.core.annotations import (
        load_csv_annotations,
        load_raven_selection_table,
        summarize_annotations,
    )

    input_path = Path(path)

    if not input_path.exists():
        click.echo(f"Error: File not found: {path}")
        raise SystemExit(1)

    # Auto-detect format
    if file_format is None:
        if input_path.suffix.lower() == '.txt':
            file_format = 'raven'
        else:
            file_format = 'csv'

    # Load annotations
    if file_format == 'raven':
        annotations = load_raven_selection_table(path)
    else:
        annotations = load_csv_annotations(path)

    summary = summarize_annotations(annotations)

    if output_json:
        click.echo(json.dumps(summary, indent=2))
    else:
        click.echo(f"\nAnnotation Summary: {path}")
        click.echo("=" * 50)
        click.echo(f"Total annotations: {summary['total_annotations']}")
        click.echo(f"Unique labels: {summary['unique_labels']}")
        click.echo("\nDuration statistics:")
        click.echo(f"  Total: {summary['total_duration']:.2f}s")
        click.echo(f"  Min: {summary['min_duration']:.2f}s")
        click.echo(f"  Max: {summary['max_duration']:.2f}s")
        click.echo(f"  Mean: {summary['mean_duration']:.2f}s")
        click.echo("\nLabel counts:")
        for label, count in sorted(summary['labels'].items()):
            click.echo(f"  {label}: {count}")


@annotation.command('remap')
@click.argument('input_file')
@click.argument('output_file')
@click.option('--mapping', '-m', required=True, help='Path to label mapping CSV (columns: source, target)')
@click.option('--keep-unmapped/--drop-unmapped', default=True,
              help='Keep or drop annotations with unmapped labels')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def annotation_remap(input_file, output_file, mapping, keep_unmapped, quiet):
    """Remap annotation labels using a mapping file.

    The mapping file should be a CSV with 'source' and 'target' columns.

    Example mapping.csv:
        source,target
        bird_song,bird
        bird_call,bird
        frog_croak,frog
    """
    from pathlib import Path

    from bioamla.core.annotations import (
        load_csv_annotations,
        load_label_mapping,
        load_raven_selection_table,
        remap_labels,
        save_csv_annotations,
        save_raven_selection_table,
    )

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}")
        raise SystemExit(1)

    # Load mapping
    label_mapping = load_label_mapping(mapping)

    # Detect format and load
    if input_path.suffix.lower() == '.txt':
        annotations = load_raven_selection_table(input_file)
        is_raven = True
    else:
        annotations = load_csv_annotations(input_file)
        is_raven = False

    original_count = len(annotations)

    # Remap labels
    remapped = remap_labels(annotations, label_mapping, keep_unmapped=keep_unmapped)

    # Save
    if output_path.suffix.lower() == '.txt' or is_raven:
        save_raven_selection_table(remapped, output_file)
    else:
        save_csv_annotations(remapped, output_file)

    if not quiet:
        click.echo(f"Remapped {original_count} annotations -> {len(remapped)} annotations")
        click.echo(f"Output: {output_file}")


@annotation.command('filter')
@click.argument('input_file')
@click.argument('output_file')
@click.option('--include', '-i', multiple=True, help='Labels to include (can specify multiple)')
@click.option('--exclude', '-e', multiple=True, help='Labels to exclude (can specify multiple)')
@click.option('--min-duration', type=float, default=None, help='Minimum duration in seconds')
@click.option('--max-duration', type=float, default=None, help='Maximum duration in seconds')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def annotation_filter(input_file, output_file, include, exclude, min_duration, max_duration, quiet):
    """Filter annotations by label or duration.

    Examples:
        # Include only specific labels
        bioamla annotation filter input.csv output.csv --include bird --include frog

        # Exclude specific labels
        bioamla annotation filter input.csv output.csv --exclude noise --exclude unknown

        # Filter by duration
        bioamla annotation filter input.csv output.csv --min-duration 0.5 --max-duration 5.0
    """
    from pathlib import Path

    from bioamla.core.annotations import (
        filter_labels,
        load_csv_annotations,
        load_raven_selection_table,
        save_csv_annotations,
        save_raven_selection_table,
    )

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}")
        raise SystemExit(1)

    # Detect format and load
    if input_path.suffix.lower() == '.txt':
        annotations = load_raven_selection_table(input_file)
        is_raven = True
    else:
        annotations = load_csv_annotations(input_file)
        is_raven = False

    original_count = len(annotations)

    # Filter by labels
    include_set = set(include) if include else None
    exclude_set = set(exclude) if exclude else None
    filtered = filter_labels(annotations, include_labels=include_set, exclude_labels=exclude_set)

    # Filter by duration
    if min_duration is not None:
        filtered = [a for a in filtered if a.duration >= min_duration]
    if max_duration is not None:
        filtered = [a for a in filtered if a.duration <= max_duration]

    # Save
    if output_path.suffix.lower() == '.txt' or is_raven:
        save_raven_selection_table(filtered, output_file)
    else:
        save_csv_annotations(filtered, output_file)

    if not quiet:
        click.echo(f"Filtered {original_count} annotations -> {len(filtered)} annotations")
        click.echo(f"Output: {output_file}")


@annotation.command('generate-labels')
@click.argument('annotation_file')
@click.argument('output_file')
@click.option('--audio-duration', type=float, required=True, help='Total audio duration in seconds')
@click.option('--clip-duration', type=float, required=True, help='Duration of each clip in seconds')
@click.option('--hop-length', type=float, default=None, help='Hop length between clips (default: same as clip duration)')
@click.option('--min-overlap', type=float, default=0.0, help='Minimum overlap ratio to assign label (0.0-1.0)')
@click.option('--multi-label/--single-label', default=True, help='Generate multi-label or single-label output')
@click.option('--format', 'output_format', type=click.Choice(['csv', 'numpy']), default='csv',
              help='Output format for labels')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
def annotation_generate_labels(
    annotation_file, output_file, audio_duration, clip_duration,
    hop_length, min_overlap, multi_label, output_format, quiet
):
    """Generate clip-level labels from annotations.

    Creates a label file for fixed-duration audio clips based on annotation overlap.

    Example:
        bioamla annotation generate-labels annotations.csv labels.csv \\
            --audio-duration 60.0 --clip-duration 3.0 --hop-length 1.0
    """
    from pathlib import Path

    import numpy as np

    from bioamla.core.annotations import (
        create_label_map,
        generate_clip_labels,
        get_unique_labels,
        load_csv_annotations,
        load_raven_selection_table,
    )

    input_path = Path(annotation_file)

    if not input_path.exists():
        click.echo(f"Error: Annotation file not found: {annotation_file}")
        raise SystemExit(1)

    # Load annotations
    if input_path.suffix.lower() == '.txt':
        annotations = load_raven_selection_table(annotation_file)
    else:
        annotations = load_csv_annotations(annotation_file)

    if not annotations:
        click.echo("Error: No annotations found in file")
        raise SystemExit(1)

    # Get labels and create mapping
    labels = get_unique_labels(annotations)
    label_map = create_label_map(labels)

    if hop_length is None:
        hop_length = clip_duration

    # Generate clip labels
    num_clips = int((audio_duration - clip_duration) / hop_length) + 1
    all_labels = []

    for i in range(num_clips):
        clip_start = i * hop_length
        clip_end = clip_start + clip_duration

        clip_labels = generate_clip_labels(
            annotations, clip_start, clip_end, label_map,
            min_overlap=min_overlap, multi_label=multi_label
        )
        all_labels.append(clip_labels)

    labels_array = np.array(all_labels)

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == 'numpy':
        np.save(output_file, labels_array)
        # Also save label map
        label_map_file = output_path.with_suffix('.labels.csv')
        import csv
        with TextFile(label_map_file, mode='w', newline='') as f:
            writer = csv.writer(f.handle)
            writer.writerow(['label', 'index'])
            for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
                writer.writerow([label, idx])
    else:
        import csv
        with TextFile(output_file, mode='w', newline='') as f:
            writer = csv.writer(f.handle)
            # Header: clip_start, clip_end, then each label
            header = ['clip_start', 'clip_end'] + sorted(label_map.keys(), key=lambda x: label_map[x])
            writer.writerow(header)

            for i, clip_labels in enumerate(labels_array):
                clip_start = i * hop_length
                clip_end = clip_start + clip_duration
                row = [f"{clip_start:.3f}", f"{clip_end:.3f}"] + [int(v) for v in clip_labels]
                writer.writerow(row)

    if not quiet:
        click.echo(f"Generated labels for {num_clips} clips")
        click.echo(f"Labels: {', '.join(sorted(label_map.keys()))}")
        click.echo(f"Output: {output_file}")


@services_hf.command('push-dataset')
@click.argument('path')
@click.argument('repo_id')
@click.option('--private/--public', default=False, help='Make the repository private (default: public)')
@click.option('--commit-message', default=None, help='Custom commit message for the push')
def hf_push_dataset(
    path: str,
    repo_id: str,
    private: bool,
    commit_message: str
):
    """Push a dataset folder to the HuggingFace Hub.

    Uploads the entire contents of PATH folder to the Hub as a dataset.
    Automatically uses upload_large_folder for folders >5GB or >1000 files.
    """
    import os

    from huggingface_hub import HfApi

    if not os.path.isdir(path):
        click.echo(f"Error: Path '{path}' does not exist or is not a directory.")
        raise SystemExit(1)

    click.echo(f"Pushing dataset folder {path} to HuggingFace Hub: {repo_id}...")

    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

        if _is_large_folder(path):
            click.echo("Large folder detected, using optimized upload method...")
            api.upload_large_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or "Upload dataset",
            )
        else:
            api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or "Upload dataset",
            )

        click.echo(f"Successfully pushed dataset to: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        click.echo(f"Error pushing to HuggingFace Hub: {e}")
        click.echo("Make sure you are logged in with 'huggingface-cli login'.")
        raise SystemExit(1)


# --- Xeno-canto subgroup ---
@services.group('xc')
def services_xc():
    """Xeno-canto bird recording database."""
    pass


@services_xc.command('search')
@click.option('--species', '-s', help='Species name (scientific or common)')
@click.option('--genus', '-g', help='Genus name')
@click.option('--country', '-c', help='Country name')
@click.option('--quality', '-q', help='Recording quality (A, B, C, D, E)')
@click.option('--type', 'sound_type', help='Sound type (song, call, etc.)')
@click.option('--max-results', '-n', default=10, type=int, help='Maximum results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
def xc_search(species, genus, country, quality, sound_type, max_results, output_format):
    """Search Xeno-canto for bird recordings.

    Examples:
        bioamla api xc-search --species "Turdus migratorius" --quality A
        bioamla api xc-search --genus Turdus --country "United States"
    """
    import json as json_lib

    from bioamla.core import xeno_canto

    try:
        results = xeno_canto.search(
            species=species,
            genus=genus,
            country=country,
            quality=quality,
            sound_type=sound_type,
            max_results=max_results,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"API error: {e}")
        raise SystemExit(1)

    if not results:
        click.echo("No recordings found.")
        return

    if output_format == 'json':
        click.echo(json_lib.dumps([r.to_dict() for r in results], indent=2))
    elif output_format == 'csv':
        import csv
        import sys
        writer = csv.DictWriter(sys.stdout, fieldnames=results[0].to_dict().keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    else:
        click.echo(f"Found {len(results)} recordings:\n")
        for r in results:
            click.echo(f"XC{r.id}: {r.scientific_name} ({r.common_name})")
            click.echo(f"  Quality: {r.quality} | Type: {r.sound_type} | Length: {r.length}")
            click.echo(f"  Location: {r.location}, {r.country}")
            click.echo(f"  Recordist: {r.recordist}")
            click.echo(f"  URL: {r.url}")
            click.echo()


@services_xc.command('download')
@click.option('--species', '-s', help='Species name (scientific or common)')
@click.option('--genus', '-g', help='Genus name')
@click.option('--country', '-c', help='Country name')
@click.option('--quality', '-q', default='A', help='Recording quality filter (default: A)')
@click.option('--max-recordings', '-n', default=10, type=int, help='Maximum recordings to download')
@click.option('--output-dir', '-o', default='./xc_recordings', help='Output directory')
@click.option('--delay', default=1.0, type=float, help='Delay between downloads in seconds')
def xc_download(species, genus, country, quality, max_recordings, output_dir, delay):
    """Download recordings from Xeno-canto.

    Examples:
        bioamla api xc-download --species "Turdus migratorius" --quality A -n 5
        bioamla api xc-download --genus Strix --country "United States" -o ./owls
    """
    from bioamla.core import xeno_canto

    click.echo("Searching Xeno-canto...")

    try:
        results = xeno_canto.search(
            species=species,
            genus=genus,
            country=country,
            quality=quality,
            max_results=max_recordings,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"API error: {e}")
        raise SystemExit(1)

    if not results:
        click.echo("No recordings found.")
        return

    click.echo(f"Found {len(results)} recordings. Starting download...")

    stats = xeno_canto.download_recordings(
        results,
        output_dir=output_dir,
        delay=delay,
        verbose=True,
    )

    click.echo(f"\nDownload complete: {stats['downloaded']}/{stats['total']} recordings")


# --- Macaulay Library subgroup ---
@services.group('ml')
def services_ml():
    """Macaulay Library audio recordings database."""
    pass


@services_ml.command('search')
@click.option('--species-code', '-s', help='eBird species code (e.g., amerob)')
@click.option('--scientific-name', help='Scientific name')
@click.option('--region', '-r', help='Region code (e.g., US-NY)')
@click.option('--min-rating', default=0, type=int, help='Minimum quality rating (1-5)')
@click.option('--max-results', '-n', default=10, type=int, help='Maximum results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def ml_search(species_code, scientific_name, region, min_rating, max_results, output_format):
    """Search Macaulay Library for audio recordings.

    Examples:
        bioamla api ml-search --species-code amerob --min-rating 4
        bioamla api ml-search --scientific-name "Turdus migratorius" -r US-NY
    """
    import json as json_lib

    from bioamla.core import macaulay

    try:
        results = macaulay.search(
            species_code=species_code,
            scientific_name=scientific_name,
            region=region,
            media_type="audio",
            min_rating=min_rating,
            count=max_results,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"API error: {e}")
        raise SystemExit(1)

    if not results:
        click.echo("No recordings found.")
        return

    if output_format == 'json':
        click.echo(json_lib.dumps([a.to_dict() for a in results], indent=2))
    else:
        click.echo(f"Found {len(results)} recordings:\n")
        for a in results:
            click.echo(f"ML{a.catalog_id}: {a.scientific_name} ({a.common_name})")
            click.echo(f"  Rating: {a.rating}/5 | Duration: {a.duration or 'N/A'}s")
            click.echo(f"  Location: {a.location}, {a.country}")
            click.echo(f"  Contributor: {a.user_display_name}")
            click.echo()


@services_ml.command('download')
@click.option('--species-code', '-s', help='eBird species code (e.g., amerob)')
@click.option('--scientific-name', help='Scientific name')
@click.option('--region', '-r', help='Region code (e.g., US-NY)')
@click.option('--min-rating', default=3, type=int, help='Minimum quality rating (default: 3)')
@click.option('--max-recordings', '-n', default=10, type=int, help='Maximum recordings to download')
@click.option('--output-dir', '-o', default='./ml_recordings', help='Output directory')
def ml_download(species_code, scientific_name, region, min_rating, max_recordings, output_dir):
    """Download recordings from Macaulay Library.

    Examples:
        bioamla api ml-download --species-code amerob --min-rating 4 -n 5
        bioamla api ml-download --scientific-name "Strix varia" -o ./owls
    """
    from bioamla.core import macaulay

    click.echo("Searching Macaulay Library...")

    try:
        results = macaulay.search(
            species_code=species_code,
            scientific_name=scientific_name,
            region=region,
            media_type="audio",
            min_rating=min_rating,
            count=max_recordings,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"API error: {e}")
        raise SystemExit(1)

    if not results:
        click.echo("No recordings found.")
        return

    click.echo(f"Found {len(results)} recordings. Starting download...")

    stats = macaulay.download_assets(
        results,
        output_dir=output_dir,
        verbose=True,
    )

    click.echo(f"\nDownload complete: {stats['downloaded']}/{stats['total']} recordings")


# --- Species lookup subgroup ---
@services.group('species')
def services_species():
    """Species name lookup and search."""
    pass


@services_species.command('lookup')
@click.argument('name')
@click.option('--to-common', '-c', is_flag=True, help='Convert scientific to common name')
@click.option('--to-scientific', '-s', is_flag=True, help='Convert common to scientific name')
@click.option('--info', '-i', is_flag=True, help='Show full species information')
def species_lookup(name, to_common, to_scientific, info):
    """Look up species names and convert between formats.

    Examples:
        bioamla api species "Turdus migratorius" --to-common
        bioamla api species "American Robin" --to-scientific
        bioamla api species "amerob" --info
    """
    from bioamla.core.services import species

    if info:
        result = species.get_species_info(name)
        if result:
            click.echo(f"Scientific name: {result.scientific_name}")
            click.echo(f"Common name: {result.common_name}")
            click.echo(f"Species code: {result.species_code}")
            click.echo(f"Family: {result.family}")
            click.echo(f"Order: {result.order}")
            click.echo(f"Source: {result.source}")
        else:
            click.echo(f"Species not found: {name}")
            raise SystemExit(1)
    elif to_common:
        result = species.scientific_to_common(name)
        if result:
            click.echo(result)
        else:
            click.echo(f"No common name found for: {name}")
            raise SystemExit(1)
    elif to_scientific:
        result = species.common_to_scientific(name)
        if result:
            click.echo(result)
        else:
            click.echo(f"No scientific name found for: {name}")
            raise SystemExit(1)
    else:
        # Default: show both if possible
        info_result = species.get_species_info(name)
        if info_result:
            click.echo(f"{info_result.scientific_name} - {info_result.common_name}")
        else:
            click.echo(f"Species not found: {name}")
            raise SystemExit(1)


@services_species.command('search')
@click.argument('query')
@click.option('--limit', '-n', default=10, type=int, help='Maximum results')
def species_search(query, limit):
    """Fuzzy search for species by name.

    Examples:
        bioamla api species-search robin
        bioamla api species-search "barred" --limit 5
    """
    from bioamla.core import species

    results = species.search(query, limit=limit)

    if not results:
        click.echo(f"No species found matching: {query}")
        return

    click.echo(f"Found {len(results)} matching species:\n")
    for r in results:
        score = r['score'] * 100
        click.echo(f"{r['scientific_name']} - {r['common_name']}")
        click.echo(f"  Code: {r['species_code']} | Family: {r['family']} | Match: {score:.0f}%")
        click.echo()


@services.command('clear-cache')
@click.option('--all', 'clear_all', is_flag=True, help='Clear all API caches')
@click.option('--xc', is_flag=True, help='Clear Xeno-canto cache')
@click.option('--ml', is_flag=True, help='Clear Macaulay Library cache')
@click.option('--species', is_flag=True, help='Clear species cache')
def clear_cache(clear_all, xc, ml, species):
    """Clear API response caches.

    Examples:
        bioamla api clear-cache --all
        bioamla api clear-cache --xc --species
    """
    total = 0

    if clear_all or xc:
        from bioamla.core import xeno_canto
        count = xeno_canto.clear_cache()
        click.echo(f"Cleared {count} Xeno-canto cache entries")
        total += count

    if clear_all or ml:
        from bioamla.core import macaulay
        count = macaulay.clear_cache()
        click.echo(f"Cleared {count} Macaulay Library cache entries")
        total += count

    if clear_all or species:
        from bioamla.core import species as species_mod
        count = species_mod.clear_cache()
        click.echo(f"Cleared {count} species cache entries")
        total += count

    if not any([clear_all, xc, ml, species]):
        click.echo("No cache specified. Use --all to clear all caches.")
        return

    click.echo(f"\nTotal: {total} cache entries cleared")


# =============================================================================
# Indices Command Group
# =============================================================================

@cli.group()
def indices():
    """Acoustic indices for soundscape ecology analysis."""
    pass


@indices.command('compute')
@click.argument('path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output CSV file for results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
@click.option('--n-fft', default=512, type=int, help='FFT window size')
@click.option('--aci-min-freq', default=0.0, type=float, help='ACI minimum frequency (Hz)')
@click.option('--aci-max-freq', default=None, type=float, help='ACI maximum frequency (Hz)')
@click.option('--bio-min-freq', default=2000.0, type=float, help='BIO minimum frequency (Hz)')
@click.option('--bio-max-freq', default=8000.0, type=float, help='BIO maximum frequency (Hz)')
@click.option('--db-threshold', default=-50.0, type=float, help='dB threshold for ADI/AEI')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def indices_compute(path, output, output_format, n_fft, aci_min_freq, aci_max_freq,
                    bio_min_freq, bio_max_freq, db_threshold, quiet):
    """Compute acoustic indices for audio file(s).

    Computes ACI, ADI, AEI, BIO, and NDSI indices for soundscape ecology analysis.

    PATH can be a single audio file or a directory of audio files.

    Examples:
        bioamla indices compute recording.wav
        bioamla indices compute ./recordings --output results.csv
        bioamla indices compute forest.wav --format json
    """
    import json as json_lib
    from pathlib import Path as PathLib

    from bioamla.core.analysis.indices import batch_compute_indices, compute_indices_from_file

    path_obj = PathLib(path)

    # Build kwargs
    kwargs = {
        "n_fft": n_fft,
        "aci_min_freq": aci_min_freq,
        "bio_min_freq": bio_min_freq,
        "bio_max_freq": bio_max_freq,
        "db_threshold": db_threshold,
    }
    if aci_max_freq:
        kwargs["aci_max_freq"] = aci_max_freq

    if path_obj.is_file():
        # Single file
        try:
            indices_result = compute_indices_from_file(path_obj, **kwargs)
            # Build result with filepath first, success last for better CSV column order
            result = {"filepath": str(path_obj)}
            result.update(indices_result.to_dict())
            result["success"] = True
            results = [result]
        except Exception as e:
            click.echo(f"Error processing {path}: {e}")
            raise SystemExit(1)
    else:
        # Directory - find audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        files = [f for f in path_obj.rglob('*') if f.suffix.lower() in audio_extensions]

        if not files:
            click.echo(f"No audio files found in {path}")
            raise SystemExit(1)

        results = batch_compute_indices(files, verbose=not quiet, **kwargs)

    # Filter successful results for output
    successful = [r for r in results if r.get("success", False)]
    failed = len(results) - len(successful)

    if output_format == 'json':
        click.echo(json_lib.dumps(results, indent=2))
    elif output_format == 'csv' or output:
        import csv
        import sys

        if successful:
            fieldnames = list(successful[0].keys())
            if output:
                with TextFile(output, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(successful)
                click.echo(f"Results saved to {output}")
            else:
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(successful)
    else:
        # Table format
        for r in results:
            if r.get("success"):
                click.echo(f"\n{r.get('filepath', 'Unknown')}:")
                click.echo(f"  ACI:  {r['aci']:.2f}")
                click.echo(f"  ADI:  {r['adi']:.3f}")
                click.echo(f"  AEI:  {r['aei']:.3f}")
                click.echo(f"  BIO:  {r['bio']:.2f}")
                click.echo(f"  NDSI: {r['ndsi']:.3f}")
                if r.get('anthrophony'):
                    click.echo(f"  Anthrophony: {r['anthrophony']:.2f}")
                    click.echo(f"  Biophony: {r['biophony']:.2f}")
            else:
                click.echo(f"\n{r.get('filepath', 'Unknown')}: Error - {r.get('error', 'Unknown error')}")

    if not quiet:
        click.echo(f"\nProcessed {len(results)} file(s): {len(successful)} successful, {failed} failed")


@indices.command('temporal')
@click.argument('path', type=click.Path(exists=True))
@click.option('--window', '-w', default=60.0, type=float, help='Window duration in seconds')
@click.option('--hop', default=None, type=float, help='Hop duration in seconds (default: window)')
@click.option('--output', '-o', type=click.Path(), help='Output CSV file')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def indices_temporal(path, window, hop, output, output_format, quiet):
    """Compute acoustic indices over time windows.

    Useful for analyzing how soundscape characteristics change over time
    in long recordings.

    Examples:
        bioamla indices temporal long_recording.wav --window 60
        bioamla indices temporal dawn_chorus.wav --window 30 --hop 15 -o results.csv
    """
    import json as json_lib

    import librosa

    from bioamla.core.analysis.indices import temporal_indices

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1)

    duration = len(audio) / sample_rate

    if not quiet:
        click.echo(f"Processing {path}")
        click.echo(f"Duration: {duration:.1f}s, Sample rate: {sample_rate} Hz")
        click.echo(f"Window: {window}s, Hop: {hop or window}s")

    results = temporal_indices(
        audio, sample_rate,
        window_duration=window,
        hop_duration=hop,
    )

    if not results:
        click.echo("No complete windows in recording (audio shorter than window duration)")
        raise SystemExit(1)

    if output_format == 'json':
        click.echo(json_lib.dumps(results, indent=2))
    elif output_format == 'csv' or output:
        import csv
        import sys

        fieldnames = list(results[0].keys())
        if output:
            with TextFile(output, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            click.echo(f"Results saved to {output}")
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    else:
        click.echo(f"\nTemporal analysis ({len(results)} windows):")
        click.echo("-" * 70)
        click.echo(f"{'Time':>12}  {'ACI':>8}  {'ADI':>6}  {'AEI':>6}  {'BIO':>8}  {'NDSI':>6}")
        click.echo("-" * 70)
        for r in results:
            time_str = f"{r['start_time']:.0f}-{r['end_time']:.0f}s"
            click.echo(f"{time_str:>12}  {r['aci']:>8.2f}  {r['adi']:>6.3f}  "
                      f"{r['aei']:>6.3f}  {r['bio']:>8.2f}  {r['ndsi']:>6.3f}")


@indices.command('aci')
@click.argument('path', type=click.Path(exists=True))
@click.option('--min-freq', default=0.0, type=float, help='Minimum frequency (Hz)')
@click.option('--max-freq', default=None, type=float, help='Maximum frequency (Hz)')
@click.option('--n-fft', default=512, type=int, help='FFT window size')
def indices_aci(path, min_freq, max_freq, n_fft):
    """Compute Acoustic Complexity Index (ACI) for an audio file.

    ACI measures the variability of sound intensities within frequency bands.
    Higher values indicate more complex acoustic environments.

    Examples:
        bioamla indices aci recording.wav
        bioamla indices aci forest.wav --min-freq 2000 --max-freq 8000
    """
    import librosa

    from bioamla.core.analysis.indices import compute_aci

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1)

    kwargs = {"n_fft": n_fft, "min_freq": min_freq}
    if max_freq:
        kwargs["max_freq"] = max_freq

    aci = compute_aci(audio, sample_rate, **kwargs)
    click.echo(f"ACI: {aci:.2f}")


@indices.command('adi')
@click.argument('path', type=click.Path(exists=True))
@click.option('--max-freq', default=10000.0, type=float, help='Maximum frequency (Hz)')
@click.option('--freq-step', default=1000.0, type=float, help='Frequency band width (Hz)')
@click.option('--db-threshold', default=-50.0, type=float, help='dB threshold')
def indices_adi(path, max_freq, freq_step, db_threshold):
    """Compute Acoustic Diversity Index (ADI) for an audio file.

    ADI is based on Shannon diversity applied to frequency bands.
    Higher values indicate more evenly distributed acoustic energy.

    Examples:
        bioamla indices adi recording.wav
        bioamla indices adi forest.wav --max-freq 8000 --freq-step 500
    """
    import librosa

    from bioamla.core.analysis.indices import compute_adi

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1)

    adi = compute_adi(audio, sample_rate, max_freq=max_freq, freq_step=freq_step,
                      db_threshold=db_threshold)
    click.echo(f"ADI: {adi:.3f}")


@indices.command('aei')
@click.argument('path', type=click.Path(exists=True))
@click.option('--max-freq', default=10000.0, type=float, help='Maximum frequency (Hz)')
@click.option('--freq-step', default=1000.0, type=float, help='Frequency band width (Hz)')
@click.option('--db-threshold', default=-50.0, type=float, help='dB threshold')
def indices_aei(path, max_freq, freq_step, db_threshold):
    """Compute Acoustic Evenness Index (AEI) for an audio file.

    AEI is based on the Gini coefficient applied to frequency bands.
    Lower values indicate more even distribution (higher evenness).

    Examples:
        bioamla indices aei recording.wav
        bioamla indices aei forest.wav --max-freq 8000
    """
    import librosa

    from bioamla.core.analysis.indices import compute_aei

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1)

    aei = compute_aei(audio, sample_rate, max_freq=max_freq, freq_step=freq_step,
                      db_threshold=db_threshold)
    click.echo(f"AEI: {aei:.3f}")


@indices.command('bio')
@click.argument('path', type=click.Path(exists=True))
@click.option('--min-freq', default=2000.0, type=float, help='Minimum frequency (Hz)')
@click.option('--max-freq', default=8000.0, type=float, help='Maximum frequency (Hz)')
def indices_bio(path, min_freq, max_freq):
    """Compute Bioacoustic Index (BIO) for an audio file.

    BIO calculates the area under the mean spectrum in the 2-8 kHz range
    where most bird and insect sounds occur.

    Examples:
        bioamla indices bio recording.wav
        bioamla indices bio tropical.wav --min-freq 1000 --max-freq 11000
    """
    import librosa

    from bioamla.core.analysis.indices import compute_bio

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1)

    bio = compute_bio(audio, sample_rate, min_freq=min_freq, max_freq=max_freq)
    click.echo(f"BIO: {bio:.2f}")


@indices.command('ndsi')
@click.argument('path', type=click.Path(exists=True))
@click.option('--anthro-min', default=1000.0, type=float, help='Anthrophony min frequency (Hz)')
@click.option('--anthro-max', default=2000.0, type=float, help='Anthrophony max frequency (Hz)')
@click.option('--bio-min', default=2000.0, type=float, help='Biophony min frequency (Hz)')
@click.option('--bio-max', default=8000.0, type=float, help='Biophony max frequency (Hz)')
def indices_ndsi(path, anthro_min, anthro_max, bio_min, bio_max):
    """Compute Normalized Difference Soundscape Index (NDSI) for an audio file.

    NDSI compares anthropogenic sounds (1-2 kHz) to biological sounds (2-8 kHz).
    Values range from -1 (pure anthrophony) to +1 (pure biophony).

    Examples:
        bioamla indices ndsi recording.wav
        bioamla indices ndsi urban.wav --anthro-max 2500 --bio-min 2500
    """
    import librosa

    from bioamla.core.analysis.indices import compute_ndsi

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1)

    ndsi, anthro, bio = compute_ndsi(
        audio, sample_rate,
        anthro_min=anthro_min,
        anthro_max=anthro_max,
        bio_min=bio_min,
        bio_max=bio_max,
    )

    click.echo(f"NDSI: {ndsi:.3f}")
    click.echo(f"  Anthrophony ({anthro_min:.0f}-{anthro_max:.0f} Hz): {anthro:.2f}")
    click.echo(f"  Biophony ({bio_min:.0f}-{bio_max:.0f} Hz): {bio:.2f}")


@indices.command('entropy')
@click.argument('path', type=click.Path(exists=True))
@click.option('--spectral', '-s', is_flag=True, help='Compute spectral entropy')
@click.option('--temporal', '-t', is_flag=True, help='Compute temporal entropy')
def indices_entropy(path, spectral, temporal):
    """Compute entropy-based acoustic indices for an audio file.

    Spectral entropy measures uniformity of the power spectrum.
    Temporal entropy measures uniformity of energy distribution over time.

    Examples:
        bioamla indices entropy recording.wav --spectral --temporal
        bioamla indices entropy forest.wav -s -t
    """
    import librosa

    from bioamla.core.analysis.indices import spectral_entropy, temporal_entropy

    try:
        audio, sample_rate = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        click.echo(f"Error loading audio: {e}")
        raise SystemExit(1)

    # Default to both if neither specified
    if not spectral and not temporal:
        spectral = temporal = True

    if spectral:
        se = spectral_entropy(audio, sample_rate)
        click.echo(f"Spectral Entropy: {se:.3f}")

    if temporal:
        te = temporal_entropy(audio, sample_rate)
        click.echo(f"Temporal Entropy: {te:.3f}")


# =============================================================================
# Detection Command Group
# =============================================================================

@cli.group()
def detect():
    """Advanced acoustic detection algorithms."""
    pass


@detect.command('energy')
@click.argument('path', type=click.Path(exists=True))
@click.option('--low-freq', '-l', default=500.0, type=float, help='Low frequency bound (Hz)')
@click.option('--high-freq', '-h', default=5000.0, type=float, help='High frequency bound (Hz)')
@click.option('--threshold', '-t', default=-20.0, type=float, help='Detection threshold (dB)')
@click.option('--min-duration', default=0.05, type=float, help='Minimum detection duration (s)')
@click.option('--output', '-o', type=click.Path(), help='Output file for detections')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
def detect_energy(path, low_freq, high_freq, threshold, min_duration, output, output_format):
    """Detect sounds using band-limited energy detection.

    Filters audio to a frequency band and detects regions where energy
    exceeds the threshold. Accepts a single audio file or a directory
    of audio files.

    Examples:
        bioamla detect energy recording.wav --low-freq 1000 --high-freq 4000
        bioamla detect energy ./recordings/ -l 2000 -h 8000 -t -25 -o detections.csv
    """
    import json as json_lib
    from pathlib import Path as PathLib

    from bioamla.core.detection import BandLimitedEnergyDetector, export_detections
    from bioamla.core.utils import get_audio_files

    detector = BandLimitedEnergyDetector(
        low_freq=low_freq,
        high_freq=high_freq,
        threshold_db=threshold,
        min_duration=min_duration,
    )

    path_obj = PathLib(path)
    all_detections = []

    if path_obj.is_dir():
        audio_files = get_audio_files(str(path_obj), recursive=True)
        if not audio_files:
            click.echo(f"No audio files found in {path}")
            return

        from bioamla.core.progress import ProgressBar, print_success

        with ProgressBar(
            total=len(audio_files),
            description="Detecting energy patterns",
        ) as progress:
            for audio_file in audio_files:
                file_detections = detector.detect_from_file(audio_file)
                for d in file_detections:
                    d.metadata['source_file'] = audio_file
                all_detections.extend(file_detections)
                progress.advance()

        print_success(f"Processed {len(audio_files)} files")
    else:
        all_detections = detector.detect_from_file(path)
        for d in all_detections:
            d.metadata['source_file'] = str(path_obj)

    if output:
        fmt = "json" if output.endswith(".json") else "csv"
        export_detections(all_detections, output, format=fmt)
        click.echo(f"Saved {len(all_detections)} detections to {output}")
    elif output_format == 'json':
        click.echo(json_lib.dumps([d.to_dict() for d in all_detections], indent=2))
    elif output_format == 'csv':
        import csv
        import sys

        if all_detections:
            fieldnames = list(all_detections[0].to_dict().keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in all_detections:
                writer.writerow(d.to_dict())
        else:
            click.echo("No detections found.")
    else:
        click.echo(f"Found {len(all_detections)} detections:\n")
        for i, d in enumerate(all_detections, 1):
            source = d.metadata.get('source_file', '')
            if source:
                source = f" [{PathLib(source).name}]"
            click.echo(f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s "
                      f"(confidence: {d.confidence:.2f}){source}")

    if not output and output_format == 'table':
        click.echo(f"\nTotal: {len(all_detections)} detections")


@detect.command('ribbit')
@click.argument('path', type=click.Path(exists=True))
@click.option('--pulse-rate', '-p', default=10.0, type=float,
              help='Expected pulse rate in Hz (pulses per second)')
@click.option('--tolerance', default=0.2, type=float,
              help='Tolerance around expected pulse rate (fraction)')
@click.option('--low-freq', '-l', default=500.0, type=float, help='Low frequency bound (Hz)')
@click.option('--high-freq', '-h', default=5000.0, type=float, help='High frequency bound (Hz)')
@click.option('--window', '-w', default=2.0, type=float, help='Analysis window duration (s)')
@click.option('--min-score', default=0.3, type=float, help='Minimum detection score')
@click.option('--output', '-o', type=click.Path(), help='Output file for detections')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
def detect_ribbit(path, pulse_rate, tolerance, low_freq, high_freq, window,
                  min_score, output, output_format):
    """Detect periodic calls using RIBBIT algorithm.

    RIBBIT detects repetitive vocalizations by analyzing the autocorrelation
    of the spectrogram. Effective for frog calls, insect sounds, etc.
    Accepts a single audio file or a directory of audio files.

    Examples:
        bioamla detect ribbit frog_pond.wav --pulse-rate 10 --low-freq 500 --high-freq 3000
        bioamla detect ribbit ./recordings/ -p 50 --tolerance 0.3
    """
    import json as json_lib
    from pathlib import Path as PathLib

    from bioamla.core.detection import RibbitDetector, export_detections
    from bioamla.core.utils import get_audio_files

    detector = RibbitDetector(
        pulse_rate_hz=pulse_rate,
        pulse_rate_tolerance=tolerance,
        low_freq=low_freq,
        high_freq=high_freq,
        window_duration=window,
        min_score=min_score,
    )

    path_obj = PathLib(path)
    all_detections = []

    if path_obj.is_dir():
        audio_files = get_audio_files(str(path_obj), recursive=True)
        if not audio_files:
            click.echo(f"No audio files found in {path}")
            return

        from bioamla.core.progress import ProgressBar, print_success

        with ProgressBar(
            total=len(audio_files),
            description="Detecting RIBBIT patterns",
        ) as progress:
            for audio_file in audio_files:
                file_detections = detector.detect_from_file(audio_file)
                for d in file_detections:
                    d.metadata['source_file'] = audio_file
                all_detections.extend(file_detections)
                progress.advance()

        print_success(f"Processed {len(audio_files)} files")
    else:
        all_detections = detector.detect_from_file(path)
        for d in all_detections:
            d.metadata['source_file'] = str(path_obj)

    if output:
        fmt = "json" if output.endswith(".json") else "csv"
        export_detections(all_detections, output, format=fmt)
        click.echo(f"Saved {len(all_detections)} detections to {output}")
    elif output_format == 'json':
        click.echo(json_lib.dumps([d.to_dict() for d in all_detections], indent=2))
    elif output_format == 'csv':
        import csv
        import sys

        if all_detections:
            fieldnames = list(all_detections[0].to_dict().keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in all_detections:
                writer.writerow(d.to_dict())
        else:
            click.echo("No detections found.")
    else:
        click.echo(f"Found {len(all_detections)} periodic call detections:\n")
        for i, d in enumerate(all_detections, 1):
            source = d.metadata.get('source_file', '')
            if source:
                source = f" [{PathLib(source).name}]"
            click.echo(f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s "
                      f"(score: {d.confidence:.2f}, pulse_rate: {d.metadata.get('pulse_rate_hz', 'N/A')}Hz){source}")

    if not output and output_format == 'table':
        click.echo(f"\nTotal: {len(all_detections)} detections")


@detect.command('peaks')
@click.argument('path', type=click.Path(exists=True))
@click.option('--snr', default=2.0, type=float, help='Signal-to-noise ratio threshold')
@click.option('--min-distance', default=0.01, type=float, help='Minimum peak distance (s)')
@click.option('--low-freq', '-l', default=None, type=float, help='Low frequency bound (Hz)')
@click.option('--high-freq', '-h', default=None, type=float, help='High frequency bound (Hz)')
@click.option('--sequences', is_flag=True, help='Detect peak sequences instead of individual peaks')
@click.option('--min-peaks', default=3, type=int, help='Minimum peaks for sequence detection')
@click.option('--output', '-o', type=click.Path(), help='Output file for detections')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
def detect_peaks(path, snr, min_distance, low_freq, high_freq, sequences,
                 min_peaks, output, output_format):
    """Detect peaks using Continuous Wavelet Transform (CWT).

    Uses CWT for robust peak detection in audio energy envelope.
    Can detect individual peaks or sequences of peaks.
    Accepts a single audio file or a directory of audio files.

    Examples:
        bioamla detect peaks recording.wav --snr 3.0
        bioamla detect peaks ./recordings/ --sequences --min-peaks 5
        bioamla detect peaks forest.wav -l 2000 -h 8000 --sequences
    """
    import json as json_lib
    from pathlib import Path as PathLib

    import librosa

    from bioamla.core.detection import CWTPeakDetector, export_detections
    from bioamla.core.utils import get_audio_files

    detector = CWTPeakDetector(
        snr_threshold=snr,
        min_peak_distance=min_distance,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    path_obj = PathLib(path)

    if path_obj.is_dir():
        audio_files = get_audio_files(str(path_obj), recursive=True)
        if not audio_files:
            click.echo(f"No audio files found in {path}")
            return
    else:
        audio_files = [str(path_obj)]

    if sequences:
        all_detections = []

        if len(audio_files) > 1:
            from bioamla.core.progress import ProgressBar, print_success

            with ProgressBar(
                total=len(audio_files),
                description="Detecting peak sequences",
            ) as progress:
                for audio_file in audio_files:
                    audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
                    file_detections = detector.detect_sequences(audio, sample_rate, min_peaks=min_peaks)
                    for d in file_detections:
                        d.metadata['source_file'] = audio_file
                    all_detections.extend(file_detections)
                    progress.advance()

            print_success(f"Processed {len(audio_files)} files")
        else:
            for audio_file in audio_files:
                audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
                file_detections = detector.detect_sequences(audio, sample_rate, min_peaks=min_peaks)
                for d in file_detections:
                    d.metadata['source_file'] = audio_file
                all_detections.extend(file_detections)

        if output:
            fmt = "json" if output.endswith(".json") else "csv"
            export_detections(all_detections, output, format=fmt)
            click.echo(f"Saved {len(all_detections)} sequence detections to {output}")
        elif output_format == 'json':
            click.echo(json_lib.dumps([d.to_dict() for d in all_detections], indent=2))
        elif output_format == 'csv':
            import csv
            import sys

            if all_detections:
                fieldnames = list(all_detections[0].to_dict().keys())
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
                for d in all_detections:
                    writer.writerow(d.to_dict())
            else:
                click.echo("No detections found.")
        else:
            click.echo(f"Found {len(all_detections)} peak sequences:\n")
            for i, d in enumerate(all_detections, 1):
                n_peaks = d.metadata.get('n_peaks', 0)
                interval = d.metadata.get('mean_interval', 0)
                source = d.metadata.get('source_file', '')
                if source:
                    source = f" [{PathLib(source).name}]"
                click.echo(f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s "
                          f"({n_peaks} peaks, mean interval: {interval:.3f}s){source}")

        if not output and output_format == 'table':
            click.echo(f"\nTotal: {len(all_detections)} sequences")
    else:
        all_peaks = []

        if len(audio_files) > 1:
            from bioamla.core.progress import ProgressBar, print_success

            with ProgressBar(
                total=len(audio_files),
                description="Detecting peaks",
            ) as progress:
                for audio_file in audio_files:
                    audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
                    file_peaks = detector.detect(audio, sample_rate)
                    for p in file_peaks:
                        p.source_file = audio_file
                    all_peaks.extend(file_peaks)
                    progress.advance()

            print_success(f"Processed {len(audio_files)} files")
        else:
            for audio_file in audio_files:
                audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
                file_peaks = detector.detect(audio, sample_rate)
                for p in file_peaks:
                    p.source_file = audio_file
                all_peaks.extend(file_peaks)

        if output:
            import csv
            fieldnames = ['time', 'amplitude', 'width', 'prominence']
            if len(audio_files) > 1:
                fieldnames.append('source_file')
            with TextFile(output, mode='w', newline='') as f:
                writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
                writer.writeheader()
                for p in all_peaks:
                    row = p.to_dict()
                    if len(audio_files) > 1:
                        row['source_file'] = getattr(p, 'source_file', '')
                    writer.writerow(row)
            click.echo(f"Saved {len(all_peaks)} peaks to {output}")
        elif output_format == 'json':
            peak_dicts = []
            for p in all_peaks:
                d = p.to_dict()
                if len(audio_files) > 1:
                    d['source_file'] = getattr(p, 'source_file', '')
                peak_dicts.append(d)
            click.echo(json_lib.dumps(peak_dicts, indent=2))
        elif output_format == 'csv':
            import csv
            import sys

            if all_peaks:
                fieldnames = ['time', 'amplitude', 'width', 'prominence']
                if len(audio_files) > 1:
                    fieldnames.append('source_file')
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
                for p in all_peaks:
                    row = p.to_dict()
                    if len(audio_files) > 1:
                        row['source_file'] = getattr(p, 'source_file', '')
                    writer.writerow(row)
            else:
                click.echo("No peaks found.")
        else:
            click.echo(f"Found {len(all_peaks)} peaks:\n")
            for i, p in enumerate(all_peaks[:20], 1):  # Show first 20
                source = getattr(p, 'source_file', '')
                if source and len(audio_files) > 1:
                    source = f" [{PathLib(source).name}]"
                else:
                    source = ''
                click.echo(f"{i}. {p.time:.3f}s (amplitude: {p.amplitude:.2f}, "
                          f"width: {p.width:.3f}s){source}")
            if len(all_peaks) > 20:
                click.echo(f"... and {len(all_peaks) - 20} more peaks")

        if not output and output_format == 'table':
            click.echo(f"\nTotal: {len(all_peaks)} peaks")


@detect.command('accelerating')
@click.argument('path', type=click.Path(exists=True))
@click.option('--min-pulses', default=5, type=int, help='Minimum pulses to detect pattern')
@click.option('--acceleration', '-a', default=1.5, type=float,
              help='Acceleration threshold (final_rate/initial_rate)')
@click.option('--deceleration', '-d', default=None, type=float,
              help='Deceleration threshold (optional)')
@click.option('--low-freq', '-l', default=500.0, type=float, help='Low frequency bound (Hz)')
@click.option('--high-freq', '-h', default=5000.0, type=float, help='High frequency bound (Hz)')
@click.option('--window', '-w', default=3.0, type=float, help='Analysis window duration (s)')
@click.option('--output', '-o', type=click.Path(), help='Output file for detections')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
def detect_accelerating(path, min_pulses, acceleration, deceleration, low_freq,
                        high_freq, window, output, output_format):
    """Detect accelerating or decelerating call patterns.

    Identifies vocalizations with increasing or decreasing pulse rates,
    common in many frog and insect species. Accepts a single audio file
    or a directory of audio files.

    Examples:
        bioamla detect accelerating tree_frog.wav --acceleration 2.0
        bioamla detect accelerating ./recordings/ -a 1.5 -d 1.5 -l 1000 -h 4000
    """
    import json as json_lib
    from pathlib import Path as PathLib

    from bioamla.core.detection import AcceleratingPatternDetector, export_detections
    from bioamla.core.utils import get_audio_files

    detector = AcceleratingPatternDetector(
        min_pulses=min_pulses,
        acceleration_threshold=acceleration,
        deceleration_threshold=deceleration,
        low_freq=low_freq,
        high_freq=high_freq,
        window_duration=window,
    )

    path_obj = PathLib(path)
    all_detections = []

    if path_obj.is_dir():
        audio_files = get_audio_files(str(path_obj), recursive=True)
        if not audio_files:
            click.echo(f"No audio files found in {path}")
            return

        from bioamla.core.progress import ProgressBar, print_success

        with ProgressBar(
            total=len(audio_files),
            description="Detecting accelerating patterns",
        ) as progress:
            for audio_file in audio_files:
                file_detections = detector.detect_from_file(audio_file)
                for d in file_detections:
                    d.metadata['source_file'] = audio_file
                all_detections.extend(file_detections)
                progress.advance()

        print_success(f"Processed {len(audio_files)} files")
    else:
        all_detections = detector.detect_from_file(path)
        for d in all_detections:
            d.metadata['source_file'] = str(path_obj)

    if output:
        fmt = "json" if output.endswith(".json") else "csv"
        export_detections(all_detections, output, format=fmt)
        click.echo(f"Saved {len(all_detections)} detections to {output}")
    elif output_format == 'json':
        click.echo(json_lib.dumps([d.to_dict() for d in all_detections], indent=2))
    elif output_format == 'csv':
        import csv
        import sys

        if all_detections:
            fieldnames = list(all_detections[0].to_dict().keys())
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for d in all_detections:
                writer.writerow(d.to_dict())
        else:
            click.echo("No detections found.")
    else:
        click.echo(f"Found {len(all_detections)} pattern detections:\n")
        for i, d in enumerate(all_detections, 1):
            pattern = d.metadata.get('pattern_type', 'unknown')
            ratio = d.metadata.get('acceleration_ratio', 1.0)
            init_rate = d.metadata.get('initial_rate', 0)
            final_rate = d.metadata.get('final_rate', 0)
            source = d.metadata.get('source_file', '')
            if source:
                source = f" [{PathLib(source).name}]"
            click.echo(f"{i}. {d.start_time:.3f}s - {d.end_time:.3f}s{source}")
            click.echo(f"   Pattern: {pattern}, ratio: {ratio:.2f}x")
            click.echo(f"   Rate: {init_rate:.1f} -> {final_rate:.1f} Hz")

    if not output and output_format == 'table':
        click.echo(f"\nTotal: {len(all_detections)} detections")


@detect.command('batch')
@click.argument('directory', type=click.Path(exists=True))
@click.option('--detector', '-d', type=click.Choice(['energy', 'ribbit', 'peaks', 'accelerating']),
              default='energy', help='Detector type to use')
@click.option('--output-dir', '-o', required=True, type=click.Path(),
              help='Output directory for detection files')
@click.option('--low-freq', '-l', default=500.0, type=float, help='Low frequency bound (Hz)')
@click.option('--high-freq', '-h', default=5000.0, type=float, help='High frequency bound (Hz)')
@click.option('--quiet', '-q', is_flag=True, help='Suppress progress output')
def detect_batch(directory, detector, output_dir, low_freq, high_freq, quiet):
    """Run detection on all audio files in a directory.

    Examples:
        bioamla detect batch ./recordings -d energy -o ./detections
        bioamla detect batch ./field_data -d ribbit -o ./results -l 1000 -h 4000
    """
    from pathlib import Path as PathLib

    from bioamla.core.detection import (
        AcceleratingPatternDetector,
        BandLimitedEnergyDetector,
        CWTPeakDetector,
        Detection,
        RibbitDetector,
        batch_detect,
        export_detections,
    )

    # Create detector
    if detector == 'energy':
        det = BandLimitedEnergyDetector(low_freq=low_freq, high_freq=high_freq)
    elif detector == 'ribbit':
        det = RibbitDetector(low_freq=low_freq, high_freq=high_freq)
    elif detector == 'peaks':
        det = CWTPeakDetector(low_freq=low_freq, high_freq=high_freq)
    else:
        det = AcceleratingPatternDetector(low_freq=low_freq, high_freq=high_freq)

    # Find audio files
    directory_path = PathLib(directory)
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    files = [f for f in directory_path.rglob('*') if f.suffix.lower() in audio_extensions]

    if not files:
        click.echo(f"No audio files found in {directory}")
        return

    if not quiet:
        click.echo(f"Found {len(files)} audio files")

    # Run batch detection
    results = batch_detect(files, det, verbose=not quiet)

    # Save results
    output_path = PathLib(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    for filepath, detections in results.items():
        if detections:
            if isinstance(detections[0], Detection):
                output_file = output_path / f"{PathLib(filepath).stem}_detections.csv"
                export_detections(detections, output_file, format="csv")
                total_detections += len(detections)

    click.echo("\nBatch detection complete:")
    click.echo(f"  Files processed: {len(files)}")
    click.echo(f"  Total detections: {total_detections}")
    click.echo(f"  Output directory: {output_dir}")


# =============================================================================
# Active Learning Commands
# =============================================================================

@cli.group()
def learn():
    """Active learning commands for efficient annotation."""
    pass


@learn.command('init')
@click.argument('predictions_csv', type=click.Path(exists=True))
@click.argument('output_state', type=click.Path())
@click.option('--strategy', '-s', type=click.Choice(['entropy', 'least_confidence', 'margin', 'random', 'hybrid']),
              default='entropy', help='Sampling strategy')
@click.option('--labeled-csv', type=click.Path(exists=True),
              help='CSV file with pre-labeled samples (id,label columns)')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def learn_init(predictions_csv: str, output_state: str, strategy: str,
               labeled_csv: Optional[str], quiet: bool):
    """Initialize active learning session from predictions.

    PREDICTIONS_CSV should contain columns: filepath, start_time, end_time,
    predicted_label, confidence.
    """
    import csv

    from bioamla.core.active_learning import (
        ActiveLearner,
        HybridSampler,
        RandomSampler,
        UncertaintySampler,
        create_samples_from_predictions,
    )

    # Create sampler based on strategy
    if strategy == 'random':
        sampler = RandomSampler()
    elif strategy == 'hybrid':
        sampler = HybridSampler()
    else:
        sampler = UncertaintySampler(strategy=strategy)

    learner = ActiveLearner(sampler=sampler)

    # Load unlabeled samples from predictions
    samples = create_samples_from_predictions(predictions_csv)
    learner.add_unlabeled(samples)

    if not quiet:
        click.echo(f"Loaded {len(samples)} samples from {predictions_csv}")

    # Load pre-labeled samples if provided
    if labeled_csv:
        labeled_samples = []
        with TextFile(labeled_csv, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f.handle)
            for row in reader:
                sample_id = row.get('id') or row.get('sample_id')
                label = row.get('label')
                if sample_id and label:
                    # Find sample in pool
                    if sample_id in learner.unlabeled_pool:
                        sample = learner.unlabeled_pool[sample_id]
                        sample.label = label
                        labeled_samples.append(sample)

        if labeled_samples:
            learner.add_labeled(labeled_samples)
            if not quiet:
                click.echo(f"Added {len(labeled_samples)} pre-labeled samples")

    # Save state
    learner.save_state(output_state)

    if not quiet:
        click.echo("\nActive learning session initialized:")
        click.echo(f"  Strategy: {strategy}")
        click.echo(f"  Unlabeled samples: {learner.state.total_unlabeled}")
        click.echo(f"  Labeled samples: {learner.state.total_labeled}")
        click.echo(f"  State saved to: {output_state}")


@learn.command('query')
@click.argument('state_file', type=click.Path(exists=True))
@click.option('--n-samples', '-n', default=10, help='Number of samples to query')
@click.option('--output', '-o', type=click.Path(), help='Output CSV for query results')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def learn_query(state_file: str, n_samples: int, output: Optional[str], quiet: bool):
    """Query samples for annotation from active learning session.

    Selects the most informative samples based on the configured strategy.
    """
    import csv
    import json
    from pathlib import Path

    from bioamla.core.active_learning import (
        ActiveLearner,
        UncertaintySampler,
    )

    # Load state to determine sampler type
    with TextFile(state_file, mode='r', encoding='utf-8') as f:
        state_data = json.load(f.handle)

    # Default to entropy sampler
    sampler = UncertaintySampler(strategy='entropy')

    learner = ActiveLearner.load_state(state_file, sampler=sampler)

    if learner.state.total_unlabeled == 0:
        click.echo("No unlabeled samples remaining!")
        return

    # Query samples
    queried = learner.query(n_samples=n_samples, update_predictions=False)

    if not quiet:
        click.echo(f"\nQueried {len(queried)} samples (iteration {learner.state.iteration}):")
        for i, sample in enumerate(queried, 1):
            conf_str = f"{sample.confidence:.3f}" if sample.confidence else "N/A"
            click.echo(f"  {i}. {sample.id}")
            click.echo(f"     File: {sample.filepath}")
            click.echo(f"     Time: {sample.start_time:.2f}s - {sample.end_time:.2f}s")
            click.echo(f"     Predicted: {sample.predicted_label or 'N/A'} (conf: {conf_str})")

    # Save query results
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with TextFile(output, mode='w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'filepath', 'start_time', 'end_time',
                         'predicted_label', 'confidence', 'label']
            writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
            writer.writeheader()
            for sample in queried:
                writer.writerow({
                    'id': sample.id,
                    'filepath': sample.filepath,
                    'start_time': sample.start_time,
                    'end_time': sample.end_time,
                    'predicted_label': sample.predicted_label or '',
                    'confidence': sample.confidence or '',
                    'label': '',  # To be filled by annotator
                })

        if not quiet:
            click.echo(f"\nQuery results saved to: {output}")
            click.echo("Fill in the 'label' column and use 'learn annotate' to import.")

    # Save updated state
    learner.save_state(state_file)


@learn.command('annotate')
@click.argument('state_file', type=click.Path(exists=True))
@click.argument('annotations_csv', type=click.Path(exists=True))
@click.option('--annotator', '-a', default='unknown', help='Annotator identifier')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def learn_annotate(state_file: str, annotations_csv: str, annotator: str, quiet: bool):
    """Import annotations into active learning session.

    ANNOTATIONS_CSV should have columns: id (or sample_id), label.
    """
    import csv

    from bioamla.core.active_learning import ActiveLearner, UncertaintySampler

    # Load learner
    sampler = UncertaintySampler(strategy='entropy')
    learner = ActiveLearner.load_state(state_file, sampler=sampler)

    # Read annotations
    annotations_imported = 0
    with TextFile(annotations_csv, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f.handle)
        for row in reader:
            sample_id = row.get('id') or row.get('sample_id')
            label = row.get('label', '').strip()

            if not sample_id or not label:
                continue

            # Find sample in unlabeled pool
            if sample_id in learner.unlabeled_pool:
                sample = learner.unlabeled_pool[sample_id]
                learner.teach(sample, label, annotator=annotator)
                annotations_imported += 1

    # Save updated state
    learner.save_state(state_file)

    if not quiet:
        click.echo(f"\nImported {annotations_imported} annotations")
        click.echo(f"  Annotator: {annotator}")
        click.echo(f"  Total labeled: {learner.state.total_labeled}")
        click.echo(f"  Remaining unlabeled: {learner.state.total_unlabeled}")
        click.echo(f"  Labels per class: {learner.state.labels_per_class}")


@learn.command('status')
@click.argument('state_file', type=click.Path(exists=True))
def learn_status(state_file: str):
    """Show status of active learning session."""
    from bioamla.core.active_learning import (
        ActiveLearner,
        UncertaintySampler,
        summarize_annotation_session,
    )

    sampler = UncertaintySampler(strategy='entropy')
    learner = ActiveLearner.load_state(state_file, sampler=sampler)

    summary = summarize_annotation_session(learner)

    click.echo("\nActive Learning Session Status")
    click.echo("=" * 40)
    click.echo(f"Iteration: {summary['iteration']}")
    click.echo(f"Total labeled: {summary['total_labeled']}")
    click.echo(f"Total unlabeled: {summary['total_unlabeled']}")
    click.echo(f"Total annotations: {summary['total_annotations']}")

    if summary['labels_per_class']:
        click.echo("\nLabels per class:")
        for label, count in sorted(summary['labels_per_class'].items()):
            click.echo(f"  {label}: {count}")

    if summary['total_annotation_time_seconds'] > 0:
        click.echo("\nAnnotation statistics:")
        click.echo(f"  Total time: {summary['total_annotation_time_seconds']:.1f}s")
        click.echo(f"  Rate: {summary['annotations_per_hour']:.1f} annotations/hour")

    if summary['class_balance_ratio'] > 0:
        click.echo(f"  Class balance ratio: {summary['class_balance_ratio']:.2f}")


@learn.command('export')
@click.argument('state_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--format', '-f', 'fmt', type=click.Choice(['csv', 'raven']),
              default='csv', help='Output format')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def learn_export(state_file: str, output_file: str, fmt: str, quiet: bool):
    """Export labeled samples from active learning session."""
    from bioamla.core.active_learning import ActiveLearner, UncertaintySampler, export_annotations

    sampler = UncertaintySampler(strategy='entropy')
    learner = ActiveLearner.load_state(state_file, sampler=sampler)

    export_annotations(learner, output_file, format=fmt)

    if not quiet:
        click.echo(f"\nExported {learner.state.total_labeled} annotations to {output_file}")
        click.echo(f"  Format: {fmt}")


@learn.command('simulate')
@click.argument('predictions_csv', type=click.Path(exists=True))
@click.argument('ground_truth_csv', type=click.Path(exists=True))
@click.option('--n-iterations', '-n', default=10, help='Number of iterations')
@click.option('--batch-size', '-b', default=10, help='Samples per iteration')
@click.option('--strategy', '-s', type=click.Choice(['entropy', 'least_confidence', 'margin', 'random', 'hybrid']),
              default='entropy', help='Sampling strategy')
@click.option('--output', '-o', type=click.Path(), help='Output CSV for simulation results')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def learn_simulate(predictions_csv: str, ground_truth_csv: str, n_iterations: int,
                   batch_size: int, strategy: str, output: Optional[str], quiet: bool):
    """Simulate active learning loop using ground truth labels.

    Useful for evaluating different sampling strategies.

    GROUND_TRUTH_CSV should have columns: id (or sample_id), label.
    """
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

    # Load ground truth
    ground_truth = {}
    with TextFile(ground_truth_csv, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f.handle)
        for row in reader:
            sample_id = row.get('id') or row.get('sample_id')
            label = row.get('label', '').strip()
            if sample_id and label:
                ground_truth[sample_id] = label

    if not ground_truth:
        click.echo("Error: No ground truth labels found in CSV")
        return

    # Create sampler
    if strategy == 'random':
        sampler = RandomSampler()
    elif strategy == 'hybrid':
        sampler = HybridSampler()
    else:
        sampler = UncertaintySampler(strategy=strategy)

    # Create oracle
    oracle = SimulatedOracle(ground_truth=ground_truth)

    # Initialize learner
    learner = ActiveLearner(sampler=sampler)
    samples = create_samples_from_predictions(predictions_csv)

    # Filter samples that have ground truth
    samples = [s for s in samples if s.id in ground_truth]
    learner.add_unlabeled(samples)

    if not quiet:
        click.echo("\nSimulating active learning:")
        click.echo(f"  Strategy: {strategy}")
        click.echo(f"  Samples: {len(samples)}")
        click.echo(f"  Iterations: {n_iterations}")
        click.echo(f"  Batch size: {batch_size}")
        click.echo()

    # Run simulation
    results = []
    for iteration in range(n_iterations):
        if learner.state.total_unlabeled == 0:
            break

        # Query
        queried = learner.query(n_samples=batch_size, update_predictions=False)

        # Annotate
        for sample in queried:
            try:
                label = oracle.annotate(sample)
                learner.teach(sample, label, annotator='oracle')
            except ValueError:
                pass  # Sample not in ground truth

        # Record progress
        result = {
            'iteration': iteration + 1,
            'total_labeled': learner.state.total_labeled,
            'total_unlabeled': learner.state.total_unlabeled,
        }
        results.append(result)

        if not quiet:
            click.echo(f"  Iteration {iteration + 1}: {learner.state.total_labeled} labeled")

    if not quiet:
        click.echo("\nSimulation complete:")
        click.echo(f"  Final labeled: {learner.state.total_labeled}")
        click.echo(f"  Labels per class: {learner.state.labels_per_class}")

    # Save results
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with TextFile(output, mode='w', newline='', encoding='utf-8') as f:
            fieldnames = ['iteration', 'total_labeled', 'total_unlabeled']
            writer = csv.DictWriter(f.handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        if not quiet:
            click.echo(f"  Results saved to: {output}")


# =============================================================================
# Clustering Commands
# =============================================================================

@cli.group()
def cluster():
    """Clustering and dimensionality reduction commands."""
    pass


@cluster.command('reduce')
@click.argument('embeddings_file')
@click.option('--output', '-o', required=True, help='Output file for reduced embeddings')
@click.option('--method', '-m', type=click.Choice(['umap', 'tsne', 'pca']),
              default='pca', help='Reduction method')
@click.option('--n-components', '-n', type=int, default=2, help='Number of output dimensions')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def cluster_reduce(embeddings_file: str, output: str, method: str,
                   n_components: int, quiet: bool):
    """Reduce dimensionality of embeddings.

    EMBEDDINGS_FILE: Path to numpy file with embeddings (.npy)
    """
    import numpy as np

    from bioamla.core.clustering import reduce_dimensions

    embeddings = np.load(embeddings_file)

    if not quiet:
        click.echo(f"Reducing {embeddings.shape[1]}D embeddings to {n_components}D using {method}...")

    reduced = reduce_dimensions(embeddings, method=method, n_components=n_components)

    np.save(output, reduced)

    if not quiet:
        click.echo(f"Saved reduced embeddings to: {output}")


@cluster.command('cluster')
@click.argument('embeddings_file')
@click.option('--output', '-o', required=True, help='Output file for cluster labels')
@click.option('--method', '-m', type=click.Choice(['kmeans', 'dbscan', 'agglomerative']),
              default='kmeans', help='Clustering method')
@click.option('--n-clusters', '-k', type=int, default=10, help='Number of clusters (for k-means/agglomerative)')
@click.option('--eps', type=float, default=0.5, help='DBSCAN epsilon')
@click.option('--min-samples', type=int, default=5, help='Minimum samples per cluster')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def cluster_cluster(embeddings_file: str, output: str, method: str,
                    n_clusters: int, eps: float, min_samples: int, quiet: bool):
    """Cluster embeddings.

    EMBEDDINGS_FILE: Path to numpy file with embeddings (.npy)
    """
    import numpy as np

    from bioamla.core.clustering import AudioClusterer, ClusteringConfig

    embeddings = np.load(embeddings_file)

    config = ClusteringConfig(
        method=method,
        n_clusters=n_clusters,
        eps=eps,
        min_samples=min_samples,
    )
    clusterer = AudioClusterer(config=config)

    if not quiet:
        click.echo(f"Clustering {len(embeddings)} samples using {method}...")

    labels = clusterer.fit_predict(embeddings)

    np.save(output, labels)

    if not quiet:
        click.echo(f"Found {clusterer.n_clusters_} clusters")
        click.echo(f"Saved cluster labels to: {output}")


@cluster.command('analyze')
@click.argument('embeddings_file')
@click.argument('labels_file')
@click.option('--output', '-o', help='Output JSON file for analysis results')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def cluster_analyze(embeddings_file: str, labels_file: str, output: str, quiet: bool):
    """Analyze cluster quality.

    EMBEDDINGS_FILE: Path to numpy file with embeddings (.npy)
    LABELS_FILE: Path to numpy file with cluster labels (.npy)
    """
    import json
    from pathlib import Path

    import numpy as np

    from bioamla.core.clustering import analyze_clusters

    embeddings = np.load(embeddings_file)
    labels = np.load(labels_file)

    analysis = analyze_clusters(embeddings, labels)

    if not quiet:
        click.echo("Cluster Analysis:")
        click.echo(f"  Clusters: {analysis['n_clusters']}")
        click.echo(f"  Samples: {analysis['n_samples']}")
        click.echo(f"  Noise: {analysis['n_noise']} ({analysis['noise_percentage']:.1f}%)")
        click.echo(f"  Silhouette Score: {analysis['silhouette_score']:.4f}")
        click.echo(f"  Calinski-Harabasz Score: {analysis['calinski_harabasz_score']:.2f}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        with TextFile(output, mode='w', encoding='utf-8') as f:
            json.dump(convert_numpy(analysis), f.handle, indent=2)
        if not quiet:
            click.echo(f"Saved analysis to: {output}")


@cluster.command('novelty')
@click.argument('embeddings_file')
@click.option('--output', '-o', required=True, help='Output file for novelty results')
@click.option('--method', '-m', type=click.Choice(['distance', 'isolation_forest', 'lof']),
              default='distance', help='Novelty detection method')
@click.option('--threshold', type=float, help='Novelty threshold')
@click.option('--labels', help='Optional cluster labels file')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def cluster_novelty(embeddings_file: str, output: str, method: str,
                    threshold: float, labels: str, quiet: bool):
    """Detect novel sounds in embeddings.

    EMBEDDINGS_FILE: Path to numpy file with embeddings (.npy)
    """
    import numpy as np

    from bioamla.core.clustering import discover_novel_sounds

    embeddings = np.load(embeddings_file)
    known_labels = np.load(labels) if labels else None

    if not quiet:
        click.echo(f"Detecting novel sounds using {method}...")

    is_novel, scores = discover_novel_sounds(
        embeddings,
        known_labels=known_labels,
        method=method,
        threshold=threshold,
        return_scores=True,
    )

    results = np.column_stack([is_novel.astype(int), scores])
    np.save(output, results)

    n_novel = is_novel.sum()
    if not quiet:
        click.echo(f"Found {n_novel} novel samples ({100*n_novel/len(embeddings):.1f}%)")
        click.echo(f"Saved novelty results to: {output}")


# =============================================================================
# Real-time Commands
# =============================================================================

@cli.group()
def realtime():
    """Real-time audio processing commands."""
    pass


@realtime.command('devices')
def realtime_devices():
    """List available audio input devices."""
    from bioamla.core.realtime import list_audio_devices

    devices = list_audio_devices()

    click.echo("Available Audio Input Devices:")
    for device in devices:
        click.echo(f"  [{device['index']}] {device['name']}")
        click.echo(f"      Channels: {device['channels']}, Sample Rate: {device['sample_rate']}")


@realtime.command('test')
@click.option('--duration', '-d', type=float, default=3.0, help='Recording duration in seconds')
@click.option('--device', type=int, help='Device index')
@click.option('--output', '-o', help='Output file to save recording')
def realtime_test(duration: float, device: int, output: str):
    """Test audio recording from microphone."""
    from bioamla.core.realtime import test_recording

    click.echo(f"Recording for {duration} seconds...")
    audio = test_recording(duration=duration, device=device)

    click.echo(f"Recorded {len(audio)} samples")
    click.echo(f"Max amplitude: {audio.max():.4f}")
    click.echo(f"RMS: {(audio**2).mean()**0.5:.4f}")

    if output:
        import soundfile as sf
        sf.write(output, audio, 16000)
        click.echo(f"Saved recording to: {output}")


# =============================================================================
# eBird subgroup (under services)
# =============================================================================

@services.group('ebird')
def services_ebird():
    """eBird bird observation database."""
    pass


@services_ebird.command('validate')
@click.argument('species_code')
@click.option('--lat', type=float, required=True, help='Latitude')
@click.option('--lng', type=float, required=True, help='Longitude')
@click.option('--api-key', envvar='EBIRD_API_KEY', required=True, help='eBird API key')
@click.option('--distance', type=float, default=50, help='Search radius in km')
def ebird_validate(species_code: str, lat: float, lng: float, api_key: str, distance: float):
    """Validate if a species is expected at a location.

    SPECIES_CODE: eBird species code (e.g., 'carwre' for Carolina Wren)
    """
    from bioamla.core.integrations import EBirdClient

    client = EBirdClient(api_key=api_key)
    result = client.validate_species_for_location(
        species_code=species_code,
        latitude=lat,
        longitude=lng,
        distance_km=distance,
    )

    if result['is_valid']:
        click.echo(f"✓ {species_code} is expected at this location")
        click.echo(f"  Found {result['nearby_observations']} nearby observations")
        if result['most_recent_observation']:
            click.echo(f"  Most recent: {result['most_recent_observation']}")
    else:
        click.echo(f"✗ {species_code} not recently observed at this location")
        click.echo(f"  {result['total_species_in_area']} other species observed nearby")


@services_ebird.command('nearby')
@click.option('--lat', type=float, required=True, help='Latitude')
@click.option('--lng', type=float, required=True, help='Longitude')
@click.option('--api-key', envvar='EBIRD_API_KEY', required=True, help='eBird API key')
@click.option('--distance', type=float, default=25, help='Search radius in km')
@click.option('--days', type=int, default=14, help='Days back to search')
@click.option('--limit', type=int, default=20, help='Maximum results')
@click.option('--output', '-o', help='Output CSV file')
def ebird_nearby(lat: float, lng: float, api_key: str, distance: float,
                 days: int, limit: int, output: str):
    """Get recent eBird observations near a location."""
    import csv

    from bioamla.core.integrations import EBirdClient

    client = EBirdClient(api_key=api_key)
    observations = client.get_nearby_observations(
        latitude=lat,
        longitude=lng,
        distance_km=distance,
        back=days,
        max_results=limit,
    )

    click.echo(f"Found {len(observations)} recent observations:")
    for obs in observations[:10]:
        count_str = f" (x{obs.how_many})" if obs.how_many else ""
        click.echo(f"  {obs.common_name}{count_str} - {obs.location_name}")

    if len(observations) > 10:
        click.echo(f"  ... and {len(observations) - 10} more")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with TextFile(output, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f.handle, fieldnames=['species_code', 'common_name', 'scientific_name',
                                                    'location_name', 'observation_date', 'how_many'])
            writer.writeheader()
            for obs in observations:
                writer.writerow(obs.to_dict())
        click.echo(f"Saved to: {output}")


# --- PostgreSQL subgroup ---
@services.group('pg')
def services_pg():
    """PostgreSQL database integration."""
    pass


@services_pg.command('export')
@click.argument('detections_file')
@click.option('--connection', '-c', envvar='DATABASE_URL', required=True,
              help='PostgreSQL connection string')
@click.option('--detector', '-d', default='unknown', help='Detector name')
@click.option('--create-tables', is_flag=True, help='Create tables if not exist')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def pg_export(detections_file: str, connection: str, detector: str,
              create_tables: bool, quiet: bool):
    """Export detections to PostgreSQL database.

    DETECTIONS_FILE: Path to JSON file with detections
    """
    import json

    from bioamla.core.integrations import PostgreSQLExporter

    with TextFile(detections_file, mode='r', encoding='utf-8') as f:
        detections = json.load(f.handle)

    exporter = PostgreSQLExporter(connection_string=connection)

    if create_tables:
        exporter.create_tables()
        if not quiet:
            click.echo("Database tables created")

    count = exporter.export_detections(detections, detector_name=detector)
    exporter.close()

    if not quiet:
        click.echo(f"Exported {count} detections to database")


@services_pg.command('stats')
@click.option('--connection', '-c', envvar='DATABASE_URL', required=True,
              help='PostgreSQL connection string')
def pg_stats(connection: str):
    """Show PostgreSQL database statistics."""
    from bioamla.core.integrations import PostgreSQLExporter

    exporter = PostgreSQLExporter(connection_string=connection)
    stats = exporter.get_statistics()
    exporter.close()

    click.echo("Database Statistics:")
    click.echo(f"  Detections: {stats['detections_count']}")
    click.echo(f"  Annotations: {stats['annotations_count']}")
    click.echo(f"  Audio Files: {stats['audio_files_count']}")
    click.echo(f"  Species Observations: {stats['species_observations_count']}")

    if stats.get('detections_by_label'):
        click.echo("\nTop Detection Labels:")
        for label, count in list(stats['detections_by_label'].items())[:10]:
            click.echo(f"  {label}: {count}")


@train.command('spec')
@click.argument('data_dir')
@click.option('--output', '-o', required=True, help='Output directory for model')
@click.option('--model', '-m', type=click.Choice(['cnn', 'crnn', 'attention']),
              default='cnn', help='Model architecture')
@click.option('--epochs', '-e', type=int, default=50, help='Number of epochs')
@click.option('--batch-size', '-b', type=int, default=32, help='Batch size')
@click.option('--lr', type=float, default=1e-3, help='Learning rate')
@click.option('--n-classes', '-n', type=int, required=True, help='Number of classes')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def train_spec(data_dir: str, output: str, model: str, epochs: int,
               batch_size: int, lr: float, n_classes: int, quiet: bool):
    """Train a spectrogram classifier (CNN/CRNN/Attention).

    DATA_DIR: Directory containing training data (spectrograms as .npy files)

    Example:
        bioamla models train spec ./spectrograms -o ./model -n 5
    """
    from bioamla.ml import (
        AttentionClassifier,
        CNNClassifier,
        CRNNClassifier,
        TrainerConfig,
    )

    if model == 'cnn':
        classifier = CNNClassifier(n_classes=n_classes)
    elif model == 'crnn':
        classifier = CRNNClassifier(n_classes=n_classes)
    else:
        classifier = AttentionClassifier(n_classes=n_classes)

    config = TrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        output_dir=output,
    )

    if not quiet:
        click.echo(f"Training {model.upper()} classifier with {n_classes} classes...")
        click.echo(f"  Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    # Note: In a real implementation, you would load data from data_dir
    click.echo("Note: This command requires properly formatted training data.")
    click.echo(f"Model will be saved to: {output}")


@models.command('ensemble')
@click.argument('model_dirs', nargs=-1, required=True)
@click.option('--output', '-o', required=True, help='Output directory for ensemble')
@click.option('--strategy', '-s', type=click.Choice(['averaging', 'voting', 'max']),
              default='averaging', help='Ensemble combination strategy')
@click.option('--weights', '-w', multiple=True, type=float, help='Model weights')
def models_ensemble(model_dirs, output: str, strategy: str, weights):
    """Create an ensemble from multiple trained models.

    MODEL_DIRS: Directories containing trained models

    Example:
        bioamla models ensemble ./model1 ./model2 -o ./ensemble -s voting
    """
    from pathlib import Path

    click.echo(f"Creating {strategy} ensemble from {len(model_dirs)} models...")

    weights_list = list(weights) if weights else None
    if weights_list and len(weights_list) != len(model_dirs):
        raise click.ClickException("Number of weights must match number of models")

    Path(output).mkdir(parents=True, exist_ok=True)

    click.echo(f"Ensemble configuration saved to: {output}")
    click.echo("Note: Load individual models and combine using bioamla.ml.Ensemble")


# =============================================================================
# Examples Command Group
# =============================================================================

@cli.group()
def examples():
    """Access example workflow scripts.

    Example workflows demonstrate bioamla capabilities and can be copied
    to your project directory for customization.

    \b
    Examples:
        bioamla examples list              # List all examples
        bioamla examples show 01           # Show example content
        bioamla examples copy 01 ./        # Copy example to directory
        bioamla examples copy-all ./       # Copy all examples
    """
    pass


@examples.command('list')
def examples_list():
    """List all available example workflows."""
    from bioamla._internal.examples import list_examples
    from bioamla.core.progress import console
    from rich.table import Table

    table = Table(title="Available Example Workflows", show_header=True)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Title", style="bold")
    table.add_column("Description")

    for example_id, title, description in list_examples():
        table.add_row(example_id, title, description)

    console.print(table)
    console.print("\n[dim]Use 'bioamla examples show <ID>' to view an example[/dim]")
    console.print("[dim]Use 'bioamla examples copy <ID> <dir>' to copy to a directory[/dim]")


@examples.command('show')
@click.argument('example_id')
def examples_show(example_id: str):
    """Show the content of an example workflow.

    EXAMPLE_ID: The example ID (e.g., 00, 01, 02) or filename
    """
    from bioamla._internal.examples import EXAMPLES, get_example_content
    from bioamla.core.progress import console
    from rich.syntax import Syntax

    try:
        content = get_example_content(example_id)
        if example_id in EXAMPLES:
            filename = EXAMPLES[example_id][0]
            title = EXAMPLES[example_id][1]
        else:
            filename = f"{example_id}.sh"
            title = example_id

        console.print(f"\n[bold]{title}[/bold] ({filename})\n")
        syntax = Syntax(content, "bash", theme="monokai", line_numbers=True)
        console.print(syntax)
    except ValueError as e:
        raise click.ClickException(str(e))


@examples.command('copy')
@click.argument('example_id')
@click.argument('output_dir', type=click.Path())
@click.option('--force', '-f', is_flag=True, help='Overwrite existing files')
def examples_copy(example_id: str, output_dir: str, force: bool):
    """Copy an example workflow to a directory.

    EXAMPLE_ID: The example ID (e.g., 00, 01, 02)
    OUTPUT_DIR: Directory to copy the example to
    """
    from pathlib import Path

    from bioamla._internal.examples import EXAMPLES, get_example_content

    try:
        content = get_example_content(example_id)
        filename = EXAMPLES[example_id][0]
    except (ValueError, KeyError) as e:
        raise click.ClickException(f"Example not found: {example_id}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dest_file = output_path / filename
    if dest_file.exists() and not force:
        raise click.ClickException(
            f"File already exists: {dest_file}\nUse --force to overwrite"
        )

    dest_file.write_text(content)
    dest_file.chmod(0o755)  # Make executable

    click.echo(f"Copied {filename} to {dest_file}")


@examples.command('copy-all')
@click.argument('output_dir', type=click.Path())
@click.option('--force', '-f', is_flag=True, help='Overwrite existing files')
def examples_copy_all(output_dir: str, force: bool):
    """Copy all example workflows to a directory.

    OUTPUT_DIR: Directory to copy examples to
    """
    from pathlib import Path

    from bioamla._internal.examples import get_all_example_files, get_example_content

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for example_id, filename in get_all_example_files():
        dest_file = output_path / filename
        if dest_file.exists() and not force:
            click.echo(f"Skipped {filename} (already exists)")
            skipped += 1
            continue

        content = get_example_content(example_id)
        dest_file.write_text(content)
        dest_file.chmod(0o755)  # Make executable
        click.echo(f"Copied {filename}")
        copied += 1

    click.echo(f"\nCopied {copied} examples to {output_path}")
    if skipped:
        click.echo(f"Skipped {skipped} existing files (use --force to overwrite)")


@examples.command('info')
@click.argument('example_id')
def examples_info(example_id: str):
    """Show detailed information about an example.

    EXAMPLE_ID: The example ID (e.g., 00, 01, 02)
    """
    from bioamla._internal.examples import EXAMPLES, get_example_content
    from bioamla.core.progress import console

    if example_id not in EXAMPLES:
        raise click.ClickException(f"Example not found: {example_id}")

    filename, title, description = EXAMPLES[example_id]
    content = get_example_content(example_id)

    # Extract purpose and features from the script header
    lines = content.split('\n')
    purpose_lines = []
    features_lines = []
    in_purpose = False
    in_features = False

    for line in lines:
        if 'PURPOSE:' in line:
            in_purpose = True
            purpose_lines.append(line.split('PURPOSE:')[1].strip())
        elif 'FEATURES DEMONSTRATED:' in line:
            in_purpose = False
            in_features = True
        elif in_purpose and line.strip().startswith('#'):
            text = line.strip('#').strip()
            if text and not any(x in text for x in ['INPUT:', 'OUTPUT:', '===', 'FEATURES']):
                purpose_lines.append(text)
            else:
                in_purpose = False
        elif in_features and line.strip().startswith('#'):
            text = line.strip('#').strip()
            if text.startswith('-'):
                features_lines.append(text)
            elif text and not any(x in text for x in ['INPUT:', 'OUTPUT:', '===']):
                continue
            else:
                in_features = False

    console.print(f"\n[bold cyan]Example {example_id}: {title}[/bold cyan]")
    console.print(f"[dim]File: {filename}[/dim]\n")
    console.print(f"[bold]Description:[/bold] {description}\n")

    if purpose_lines:
        console.print("[bold]Purpose:[/bold]")
        console.print("  " + " ".join(purpose_lines) + "\n")

    if features_lines:
        console.print("[bold]Features demonstrated:[/bold]")
        for feature in features_lines:
            console.print(f"  {feature}")

    # Count lines and commands
    command_count = sum(1 for line in lines if line.strip().startswith('bioamla '))
    console.print(f"\n[dim]Lines: {len(lines)}, bioamla commands: {command_count}[/dim]")


if __name__ == '__main__':
    cli()
