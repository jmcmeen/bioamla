"""System information commands (version, devices, dependencies, config schemas)."""

import click


@click.group()
def system() -> None:
    """System information commands (version, devices, dependencies, config schemas)."""
    pass


@system.command("version")
def system_version() -> None:
    """Show bioamla version and environment information."""
    from bioamla.cli.console import console
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    version_data = run_operation("system.version")
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
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    devices_data = run_operation("system.devices")

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
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result, json_mode

    if json_mode() and do_install and not yes:
        raise click.UsageError("--install with --json needs --yes (there is no prompt).")

    if json_mode():
        emit_result(run_operation("system.deps-install" if do_install else "system.deps"))
        return
    report = run_operation("system.deps")

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
        installed = run_operation("system.deps-install")
        console.print(f"[green]{installed['installed'] or 'Nothing was missing.'}[/green]")
        console.print("\n[bold]Verifying installation...[/bold]")
        for dep in installed["report"].dependencies:
            if dep.installed:
                console.print(f"[green]✓[/green] {dep.name}")
            else:
                console.print(f"[red]✗[/red] {dep.name}")


@system.command("commands")
@click.argument("path", nargs=-1)
@click.pass_context
def system_commands(ctx: click.Context, path: tuple[str, ...]) -> None:
    """List every command, or describe the commands under PATH (e.g. `detect energy`).

    With --json: every command's help, arguments and options (type, choices,
    default, required, flags), plus each option's BIOAMLA_* variable and settings
    key. Scripts and agent tools use it to learn the CLI without parsing --help.

    \b
    Examples:
        bioamla system commands
        bioamla system commands detect energy
        bioamla --json system commands > commands.json
    """
    from bioamla.cli.catalog import command_catalog
    from bioamla.cli.console import print_kv, print_table
    from bioamla.cli.output import emit_result
    from bioamla.exceptions import NotFoundError

    try:
        catalog = command_catalog(ctx.find_root().command, list(path))
    except NotFoundError as e:
        raise click.UsageError(str(e)) from e
    if emit_result(catalog):
        return
    commands = catalog["commands"]
    leaf = commands[0] if path and commands and commands[0]["kind"] == "command" else None
    if leaf is None:
        rows = [[f"bioamla {c['path']}", c["summary"]] for c in commands if c["kind"] == "command"]
        print_table(f"{len(rows)} commands", ["command", "what it does"], rows)
        return
    print_kv("Command", f"bioamla {leaf['path']}")
    click.echo(f"\n{leaf['help']}\n")
    if leaf["arguments"]:
        print_table("Arguments", ["name", "type", "required"],
                    [[a["name"], a["type"], "yes" if a["required"] else ""]
                     for a in leaf["arguments"]])  # fmt: skip
    print_table(
        "Options",
        ["flags", "type", "required", "default", "setting / env"],
        [[", ".join(o["flags"]), o["type"] + (f" {o['choices']}" if o.get("choices") else ""),
          "yes" if o["required"] else "",
          "" if o["default"] in (None, []) else repr(o["default"]),
          f"{o.get('setting') or ''}\n{o.get('env') or ''}".strip()]
         for o in leaf["options"]],
    )  # fmt: skip


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
    from bioamla.cli.operations import run_operation
    from bioamla.cli.output import emit_result

    if export_all:
        if name or not output:
            raise click.UsageError("--all takes no NAME and needs --output DIRECTORY.")
        exported = run_operation("system.export-schemas", output_dir=output)
        for skipped in exported["skipped"]:
            click.echo(f"Skipped {skipped['name']}: {skipped['reason']}", err=True)
        for path in exported["written"]:
            click.echo(f"Wrote {path}")
        emit_result(exported)
        return

    result = run_operation("system.schema", name=name, output=output)
    if emit_result(result):
        return
    if not name:
        for config_name, target in result["configs"].items():
            click.echo(f"{config_name:<26} {target}")
    elif output:
        click.echo(f"Wrote {output}")
    else:
        import json

        click.echo(json.dumps(result["schema"], indent=2) + "\n", nl=False)
