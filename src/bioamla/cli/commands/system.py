"""System information commands (version, devices, dependencies, config schemas)."""

import click

from bioamla.exceptions import BioamlaError


@click.group()
def system() -> None:
    """System information commands (version, devices, dependencies, config schemas)."""
    pass


@system.command("version")
def system_version() -> None:
    """Show bioamla version and environment information."""
    from bioamla.cli.console import console
    from bioamla.system import util

    try:
        version_data = util.get_version()
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    from bioamla.cli.output import emit_result

    if emit_result(version_data):
        return

    console.print("\n[bold]BioAMLA Version Information[/bold]\n")
    console.print(f"  bioamla:  {version_data.bioamla_version}")
    console.print(f"  Python:   {version_data.python_version.split()[0]}")
    console.print(f"  Platform: {version_data.platform}")

    if version_data.pytorch_version:
        console.print(f"  PyTorch:  {version_data.pytorch_version}")
    if version_data.cuda_version:
        console.print(f"  CUDA:     {version_data.cuda_version}")

    console.print()


@system.command("devices")
def system_devices() -> None:
    """Show available compute devices (GPU, MPS, CPU)."""
    from bioamla.cli.console import console
    from bioamla.system import util

    try:
        devices_data = util.get_device_info()
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    from bioamla.cli.output import emit_result

    if emit_result(devices_data):
        return

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
        console.print(
            f"[green]CUDA available[/green] ({cuda_count} GPU{'s' if cuda_count > 1 else ''})"
        )
    elif devices_data.mps_available:
        console.print("[green]Apple MPS available[/green]")
    else:
        console.print("[yellow]No GPU acceleration available - using CPU[/yellow]")

    console.print()


@system.command("deps")
@click.option("--install", "do_install", is_flag=True, help="Install missing system dependencies")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def system_deps(do_install: bool, yes: bool) -> None:
    """Check or install system dependencies (FFmpeg, libsndfile, PortAudio).

    These system libraries are required for full bioamla functionality:

    \b
      - FFmpeg: Audio format conversion (MP3, FLAC, etc.)
      - libsndfile: Audio file I/O
      - PortAudio: Real-time audio recording

    \b
    Examples:
        bioamla system deps              # Check dependencies
        bioamla system deps --install    # Install missing dependencies
        bioamla system deps --install -y # Install without confirmation
    """
    from bioamla.cli.console import console
    from bioamla.cli.output import emit_result, json_mode
    from bioamla.system import dependency

    if json_mode() and do_install and not yes:
        raise click.UsageError("--install with --json needs --yes (there is no prompt).")

    try:
        report = dependency.check_all()
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e

    if json_mode():
        if not do_install:
            emit_result(report)
            return
        if report.all_installed:  # nothing to do, as in the normal path
            emit_result({"installed": None, "report": report})
            return
        try:
            message = dependency.install(report.os_type)
        except BioamlaError as e:
            raise click.ClickException(str(e)) from e
        emit_result({"installed": message, "report": dependency.check_all()})
        return

    console.print("\n[bold]System Dependencies[/bold]")
    console.print(f"[dim]Detected OS: {report.os_type}[/dim]\n")

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
        console.print("[dim]Or run: bioamla system deps --install[/dim]")

    if do_install:
        console.print()
        if not yes:
            if not click.confirm("Install missing system dependencies?"):
                click.echo("Aborted.")
                return

        console.print("\n[bold]Installing dependencies...[/bold]")
        try:
            message = dependency.install(report.os_type)
        except BioamlaError as e:
            raise click.ClickException(str(e)) from e

        console.print(f"[green]{message}[/green]")
        console.print("\n[bold]Verifying installation...[/bold]")
        verify = dependency.check_all()
        for dep in verify.dependencies:
            if dep.installed:
                console.print(f"[green]✓[/green] {dep.name}")
            else:
                console.print(f"[red]✗[/red] {dep.name}")


@system.command("schema")
@click.argument("name", required=False)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Write to this file (with --all, this directory) instead of stdout",
)
@click.option("--all", "export_all", is_flag=True, help="Export every config into --output")
def system_schema(name: str | None, output: str | None, export_all: bool) -> None:
    """Print the JSON Schema of a config, or list the configs.

    With no NAME, lists the configs that have a schema. Point an editor's TOML
    or JSON schema support at the output to get completion and validation.
    """
    import json
    from pathlib import Path

    from bioamla.cli.output import emit_result
    from bioamla.common.config import CONFIG_CLASSES, config_schema
    from bioamla.exceptions import DependencyError

    if export_all:
        if name or not output:
            raise click.UsageError("--all takes no NAME and needs --output DIRECTORY.")
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        skipped: list[dict[str, str]] = []
        for config_name in CONFIG_CLASSES:
            try:
                schema = config_schema(config_name)
            except DependencyError as e:
                skipped.append({"name": config_name, "reason": str(e)})
                click.echo(f"Skipped {config_name}: {e}", err=True)
                continue
            path = out_dir / f"{config_name}.schema.json"
            path.write_text(json.dumps(schema, indent=2) + "\n")
            written.append(str(path))
            click.echo(f"Wrote {path}")
        emit_result({"written": written, "skipped": skipped})
        return

    if not name:
        if emit_result({"configs": {n: t.replace(":", ".") for n, t in CONFIG_CLASSES.items()}}):
            return
        for config_name, target in CONFIG_CLASSES.items():
            click.echo(f"{config_name:<26} {target.replace(':', '.')}")
        return

    try:
        schema = config_schema(name)
    except BioamlaError as e:
        raise click.ClickException(str(e)) from e
    text = json.dumps(schema, indent=2) + "\n"
    if output:
        Path(output).write_text(text)
        if emit_result({"name": name, "output": output}):
            return
        click.echo(f"Wrote {output}")
    elif not emit_result({"name": name, "schema": schema}):
        click.echo(text, nl=False)
