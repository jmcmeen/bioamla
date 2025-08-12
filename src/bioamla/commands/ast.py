"""
AST Project Creation Command
============================

Command-line tool for creating new Audio Spectrogram Transformer (AST) projects.
This utility sets up a new AST project directory with default configuration files
needed for training and inference operations.

Usage:
    ast FILEPATH

Examples:
    ast my_ast_project              # Create AST project in ./my_ast_project
    ast /path/to/new_project        # Create AST project at specified path
"""

import click
from novus_pytils.files import directory_exists, create_directory, copy_files
from novus_pytils.config.yaml import get_yaml_files
from pathlib import Path

module_dir = Path(__file__).parent
config_dir = module_dir.joinpath("config")

@click.command()
@click.argument('filepath')
def main(filepath: str):
    """
    Create a new AST project directory with configuration templates.
    
    Creates a new directory at the specified path and copies default AST
    configuration files (YAML templates) into it. These configuration files
    can be customized for specific training and inference tasks.
    
    Args:
        filepath (str): Path where the new AST project directory should be created.
                       Must not already exist as a directory.
    
    Raises:
        ValueError: If the specified directory already exists.
    """
    if directory_exists(filepath):
        raise ValueError("Existing directory")

    create_directory(filepath)
    config_files = get_yaml_files(config_dir)
    copy_files(config_files, filepath)

    click.echo(f"AST project created at {filepath}")

if __name__ == '__main__':
    main()