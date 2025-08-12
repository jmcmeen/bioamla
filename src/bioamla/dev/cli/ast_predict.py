import click
from bioamla.core.ast import wav_ast_inference


@click.command()
@click.argument('filepath')
@click.argument('model_path')
@click.argument('sample_rate')
def main(filepath, model_path, sample_rate):
    prediction = wav_ast_inference(filepath, model_path, int(sample_rate))
    click.echo(f"{prediction}")

if __name__ == '__main__':
    main()