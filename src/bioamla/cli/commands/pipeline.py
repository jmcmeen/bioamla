"""Pipeline execution commands."""

import click


@click.group()
def pipeline():
    """Pipeline execution commands."""
    pass


@pipeline.command("run")
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file")
@click.option("--output", "-o", help="Output directory")
@click.option("--dry-run", is_flag=True, help="Show what would be executed without running")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def pipeline_run(pipeline_file, config, output, dry_run, verbose):
    """Execute a pipeline from a YAML or JSON configuration file."""
    from pathlib import Path

    from bioamla.core.pipeline import Pipeline

    try:
        pipeline_obj = Pipeline.load(pipeline_file)

        if config:
            pipeline_obj.load_config(config)

        if output:
            pipeline_obj.output_dir = Path(output)

        if dry_run:
            click.echo("Pipeline steps:")
            for i, step in enumerate(pipeline_obj.steps, 1):
                click.echo(f"  {i}. {step.name}: {step.description}")
            click.echo(f"\nTotal: {len(pipeline_obj.steps)} steps")
            return

        click.echo(f"Running pipeline: {pipeline_obj.name}")
        result = pipeline_obj.run(verbose=verbose)

        if result.success:
            click.echo(f"Pipeline completed successfully in {result.elapsed_time:.2f}s")
        else:
            click.echo(f"Pipeline failed at step '{result.failed_step}': {result.error}")
            raise SystemExit(1)

    except Exception as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from e


@pipeline.command("validate")
@click.argument("pipeline_file", type=click.Path(exists=True))
def pipeline_validate(pipeline_file):
    """Validate a pipeline configuration file."""
    from bioamla.core.pipeline import Pipeline

    try:
        pipeline_obj = Pipeline.load(pipeline_file)
        errors = pipeline_obj.validate()

        if errors:
            click.echo("Validation errors:")
            for error in errors:
                click.echo(f"  - {error}")
            raise SystemExit(1)
        else:
            click.echo(f"Pipeline '{pipeline_obj.name}' is valid")
            click.echo(f"  Steps: {len(pipeline_obj.steps)}")

    except Exception as e:
        click.echo(f"Error loading pipeline: {e}")
        raise SystemExit(1) from e


@pipeline.command("list")
def pipeline_list():
    """List available pipeline templates."""
    from bioamla.core.pipeline import list_templates

    templates = list_templates()

    if not templates:
        click.echo("No pipeline templates available")
        return

    click.echo("Available pipeline templates:")
    for name, description in templates:
        click.echo(f"  {name}: {description}")


@pipeline.command("init")
@click.argument("output_file", type=click.Path())
@click.option("--template", "-t", default="basic", help="Template to use")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file")
def pipeline_init(output_file, template, force):
    """Create a new pipeline configuration file from a template."""
    from pathlib import Path

    from bioamla.core.pipeline import create_from_template

    output_path = Path(output_file)

    if output_path.exists() and not force:
        click.echo(f"File already exists: {output_file}")
        click.echo("Use --force to overwrite")
        raise SystemExit(1)

    try:
        create_from_template(template, output_path)
        click.echo(f"Created pipeline configuration: {output_file}")

    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from e
