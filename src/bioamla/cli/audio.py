import click
from novus_pytils.audio.files import find_audio_files

@click.command()
@click.argument('filepath')
def main(filepath):
    metadata = find_audio_files(filepath)
    click.echo(f"{metadata}")

if __name__ == '__main__':
    main()