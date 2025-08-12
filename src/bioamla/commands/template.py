# /src/bioamla/commands/template.py
# This is a template for creating new command scripts in the bioamla package.
import click

@click.command()
@click.argument('option')
def main(option: str):
    click.echo(f"{option} command executed.")

if __name__ == '__main__':
    main()