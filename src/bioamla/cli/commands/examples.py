"""Access example pipeline scripts."""

import click


@click.group()
def examples():
    """Access example pipeline scripts.

    Example pipelines demonstrate bioamla capabilities and can be copied
    to your project directory for customization.

    \b
    Examples:
        bioamla examples list              # List all examples
        bioamla examples show 01           # Show example content
        bioamla examples copy 01 ./        # Copy example to directory
        bioamla examples copy-all ./       # Copy all examples
    """
    pass


@examples.command("list")
def examples_list():
    """List all available example pipelines."""
    from rich.table import Table

    from bioamla._internal.examples import list_examples
    from bioamla.cli.progress import console

    table = Table(title="Available Example Workflows", show_header=True)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Title", style="bold")
    table.add_column("Description")

    for example_id, title, description in list_examples():
        table.add_row(example_id, title, description)

    console.print(table)
    console.print("\n[dim]Use 'bioamla examples show <ID>' to view an example[/dim]")
    console.print("[dim]Use 'bioamla examples copy <ID> <dir>' to copy to a directory[/dim]")


@examples.command("show")
@click.argument("example_id")
def examples_show(example_id: str):
    """Show the content of an example pipeline."""
    from rich.syntax import Syntax

    from bioamla._internal.examples import EXAMPLES, get_example_content
    from bioamla.cli.progress import console

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
        raise click.ClickException(str(e)) from e


@examples.command("copy")
@click.argument("example_id")
@click.argument("output_dir", type=click.Path())
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def examples_copy(example_id: str, output_dir: str, force: bool):
    """Copy an example pipeline to a directory."""
    from pathlib import Path

    from bioamla._internal.examples import EXAMPLES, get_example_content

    try:
        content = get_example_content(example_id)
        filename = EXAMPLES[example_id][0]
    except (ValueError, KeyError) as e:
        raise click.ClickException(f"Example not found: {example_id}") from e

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dest_file = output_path / filename
    if dest_file.exists() and not force:
        raise click.ClickException(f"File already exists: {dest_file}\nUse --force to overwrite")

    dest_file.write_text(content)
    dest_file.chmod(0o755)

    click.echo(f"Copied {filename} to {dest_file}")


@examples.command("copy-all")
@click.argument("output_dir", type=click.Path())
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def examples_copy_all(output_dir: str, force: bool):
    """Copy all example pipelines to a directory."""
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
        dest_file.chmod(0o755)
        click.echo(f"Copied {filename}")
        copied += 1

    click.echo(f"\nCopied {copied} examples to {output_path}")
    if skipped:
        click.echo(f"Skipped {skipped} existing files (use --force to overwrite)")


@examples.command("info")
@click.argument("example_id")
def examples_info(example_id: str):
    """Show detailed information about an example."""
    from bioamla._internal.examples import EXAMPLES, get_example_content
    from bioamla.cli.progress import console

    if example_id not in EXAMPLES:
        raise click.ClickException(f"Example not found: {example_id}")

    filename, title, description = EXAMPLES[example_id]
    content = get_example_content(example_id)

    lines = content.split("\n")
    purpose_lines = []
    features_lines = []
    in_purpose = False
    in_features = False

    for line in lines:
        if "PURPOSE:" in line:
            in_purpose = True
            purpose_lines.append(line.split("PURPOSE:")[1].strip())
        elif "FEATURES DEMONSTRATED:" in line:
            in_purpose = False
            in_features = True
        elif in_purpose and line.strip().startswith("#"):
            text = line.strip("#").strip()
            if text and not any(x in text for x in ["INPUT:", "OUTPUT:", "===", "FEATURES"]):
                purpose_lines.append(text)
            else:
                in_purpose = False
        elif in_features and line.strip().startswith("#"):
            text = line.strip("#").strip()
            if text.startswith("-"):
                features_lines.append(text)
            elif text and not any(x in text for x in ["INPUT:", "OUTPUT:", "==="]):
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

    command_count = sum(1 for line in lines if line.strip().startswith("bioamla "))
    console.print(f"\n[dim]Lines: {len(lines)}, bioamla commands: {command_count}[/dim]")
