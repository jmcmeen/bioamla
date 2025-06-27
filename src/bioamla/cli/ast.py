import click
from novus_pytils.files import directory_exists, create_directory, copy_directory
from pathlib import Path

module_dir = Path(__file__).parent
config_dir = module_dir.joinpath("config")

@click.command()
@click.argument('filepath')
def main(filepath):
    if directory_exists(filepath):
        raise ValueError("Existing directory")

    create_directory(filepath)
    # click.echo(module_dir)
    copy_directory(config_dir, filepath)

    click.echo(f"AST project created at {filepath}")

if __name__ == '__main__':
    main()