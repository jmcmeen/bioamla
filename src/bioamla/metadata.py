import click

@click.command()
@click.option('--name', default='World', help='Name to greet.')
def main(name):
    """Simple CLI tool to greet a name."""
    click.echo(f"Hello, {name}!")

if __name__ == '__main__':
    main()