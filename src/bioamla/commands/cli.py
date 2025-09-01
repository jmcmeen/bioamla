import click


@click.group()
def cli():
    """Bioamla CLI"""
    pass

@cli.command()
def devices():
    """
    Display comprehensive device information including CUDA and GPU details.
    
    Retrieves and displays information about available compute devices,
    focusing on CUDA-capable GPUs that can be used for machine learning
    inference and training tasks.
    """
    from bioamla.core.diagnostics import get_device_info
    device_info = get_device_info()
    
    click.echo("Devices:")
    click.echo(f'CUDA available: {device_info["cuda_available"]}')
    click.echo(f'Current device: {device_info["current_device"]}')
    click.echo(f'Device count: {device_info["device_count"]}')

    for device in device_info['devices']:
        click.echo(f'  - Index: {device["index"]}, Name: {device["name"]}')

@cli.command()
@click.argument('url', required=True)
@click.argument('output_dir', required=False, default='.')
def download(url: str, output_dir: str):
    """
    Download a file from the specified URL to the target directory.
    
    Downloads a file from the given URL and saves it to the specified output
    directory. If no output directory is provided, downloads to the current
    working directory.
    
    Args:
        url (str): The URL of the file to download
        output_dir (str): Directory where the file should be saved.
                         Defaults to current directory if not specified.
    """
    from novus_pytils.files import download_file
    import os
    
    if output_dir == '.':
        output_dir = os.getcwd()
        
    download_file(url, output_dir)

@cli.command()
@click.argument('filepath', required=False, default='.')
def audio(filepath: str):
    """
    Display audio files from a specified directory.

    Args:
        filepath (str): The path to the directory to search for audio files.
                        Defaults to the current directory if not provided.
    """
    from novus_pytils.audio import get_audio_files
    try:
        if filepath == '.':
            import os
            filepath = os.getcwd()
        audio_files = get_audio_files(filepath)
        if audio_files:
            for file in audio_files:
                click.echo(file)
        else:
            click.echo("No audio files found in the specified directory.")
    except Exception as e:
        click.echo(f"An error occurred: {e}")
     
@cli.command()
@click.argument('file_path')
@click.argument('output_path', required=False, default='.')
def unzip(file_path: str, output_path: str):
    """
    Extract a ZIP archive to the specified output directory.
    
    Extracts the contents of a ZIP file to the target directory.
    If no output path is specified, extracts to the current working directory.
    
    Args:
        file_path (str): Path to the ZIP file to extract
        output_path (str): Directory where the ZIP contents should be extracted.
                          Defaults to current directory if not specified.
    """
    from novus_pytils.compression import extract_zip_file
    if output_path == '.':
        import os
        output_path = os.getcwd()
  
    extract_zip_file(file_path, output_path)   

@cli.command()
def version():
    """
    Display the current version of the bioamla package.
    
    This command retrieves and displays the version information
    for the installed bioamla package.
    """
    from bioamla.core.diagnostics import get_bioamla_version
    click.echo(f"bioamla v{get_bioamla_version()}")  
     
@cli.command()
@click.argument('filepath')
def ast(filepath: str):
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
    from novus_pytils.files import directory_exists, create_directory, copy_files
    from novus_pytils.text.yaml import get_yaml_files
    from pathlib import Path
    
    module_dir = Path(__file__).parent
    config_dir = module_dir.joinpath("../config")

    if directory_exists(filepath):
        raise ValueError("Existing directory")

    create_directory(filepath)
    config_files = get_yaml_files(str(config_dir))
    
    copy_files(config_files, filepath)

    click.echo(f"AST project created at {filepath}")
        
if __name__ == '__main__':
    cli()