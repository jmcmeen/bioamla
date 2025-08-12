# src/bioamla/commands/version.py
# Command to display the current version of the bioamla package.
# TODO: Make small splash screen with version and some info.
# TODO: Update the command to flag for displaying all installed packages and their versions.
import click

@click.command()
def main():
    from bioamla.controllers.version import get_bioamla_version
    click.echo(f"bioamla v{get_bioamla_version()}")

if __name__ == '__main__':
    main()
``