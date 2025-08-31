"""
Command Template
================

Template file for creating new command scripts in the bioamla package.
This file serves as a starting point for implementing new command-line
tools with consistent structure and documentation patterns.

Usage:
    template OPTION

Examples:
    template test               # Execute template command with 'test' option
    template example            # Execute template command with 'example' option

Template Structure:
    - Module-level docstring with comprehensive description
    - Import statements (click for CLI functionality)
    - Click command decorator and argument/option definitions
    - Main function with proper type hints and docstring
    - Parameter documentation following established patterns
    - Conditional main execution block
"""

import click

@click.command()
@click.argument('option')
def main(option: str):
    """
    Template command function that demonstrates basic CLI structure.
    
    This function serves as a template for implementing new commands in the
    bioamla package. It shows the basic pattern of accepting arguments and
    producing output using the Click library.
    
    Args:
        option (str): A string argument that will be echoed back to the user
                     as part of the command execution message.
    """
    click.echo(f"{option} command executed.")

if __name__ == '__main__':
    main()