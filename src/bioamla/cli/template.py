import click

@click.command()
@click.argument('filepath')
def main(filepath):
    click.echo(f"{filepath}")

if __name__ == '__main__':
    main()