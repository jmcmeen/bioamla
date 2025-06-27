import click
from novus_pytils.wave import get_wav_file_metadata

@click.command()
@click.argument('filepath')
def print_metadata(filepath):
    metadata = get_wav_file_metadata(filepath)
    click.echo(f"{metadata}")

if __name__ == '__main__':
    print_metadata()