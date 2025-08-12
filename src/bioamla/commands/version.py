"""
Version Information Command
==========================

Command-line tool for displaying the current version of the bioamla package.
This utility provides version information for troubleshooting and support.

Usage:
    version

Examples:
    version    # Display current bioamla version

Future enhancements:
- Add splash screen with additional package information
- Option to display all installed dependencies and their versions
"""
import click

@click.command()
def main():
    """
    Display the current version of the bioamla package.
    
    This command retrieves and displays the version information
    for the installed bioamla package.
    """
    from bioamla.controllers.diagnostics import get_bioamla_version
    click.echo(f"bioamla v{get_bioamla_version()}")

if __name__ == '__main__':
    main()
