import click
from novus_pytils.audio.files import get_audio_files

@click.command()
@click.argument('filepath')
def main(filepath):
    metadata = get_audio_files(filepath)
    click.echo(f"{metadata}")

if __name__ == '__main__':
    main()