import click
from bioamla.controllers.audio import get_audio_files_from_directory

@click.command()
@click.argument('filepath', required=False, default='.')
def main(filepath: str):
    """
    Command-line interface to get audio files from a specified directory.

    Args:
        filepath (str): The path to the directory to search for audio files.
                        Defaults to the current directory if not provided.
    """
    try:
        if filepath == '.':
            import os
            filepath = os.getcwd()
        audio_files = get_audio_files_from_directory(filepath)
        if audio_files:
            click.echo("Audio files found:")
            for file in audio_files:
                click.echo(file)
        else:
            click.echo("No audio files found in the specified directory.")
    except Exception as e:
        click.echo(f"An error occurred: {e}")

if __name__ == '__main__':
    main()