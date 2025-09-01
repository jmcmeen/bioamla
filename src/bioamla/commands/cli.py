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
    Command-line interface to get audio files from a specified directory.

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
            click.echo("Audio files found:")
            for file in audio_files:
                click.echo(file)
        else:
            click.echo("No audio files found in the specified directory.")
    except Exception as e:
        click.echo(f"An error occurred: {e}")
        
if __name__ == '__main__':
    cli()