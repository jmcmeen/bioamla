import click
from novus_pytils.files import directory_exists, create_directory, copy_directory

@click.command()
@click.argument('filepath')
def main(filepath):
    if directory_exists(filepath):
        raise ValueError("Existing directory")

    create_directory(filepath)
    copy_directory("../config/", filepath)

    click.echo(f"AST project created at {filepath}")

if __name__ == '__main__':
    main()