"""
AST Model Prediction Command
============================

Command-line tool for running Audio Spectrogram Transformer (AST) model
predictions on single audio files. This utility loads a pre-trained AST
model and performs inference on the provided audio file.

Usage:
    ast-predict FILEPATH MODEL_PATH SAMPLE_RATE

Examples:
    ast-predict audio.wav ./model 16000
    ast-predict /path/to/audio.wav MIT/ast-finetuned-audioset-10-10-0.4593 16000
"""

import click
from bioamla.core.ast import wav_ast_inference


@click.command()
@click.argument('filepath')
@click.argument('model_path')
@click.argument('sample_rate')
def main(filepath, model_path, sample_rate):
    """
    Perform AST model prediction on a single audio file.
    
    Args:
        filepath: Path to the audio file to classify
        model_path: Path to the pre-trained AST model
        sample_rate: Target sample rate for audio preprocessing
    """
    prediction = wav_ast_inference(filepath, model_path, int(sample_rate))
    click.echo(f"{prediction}")

if __name__ == '__main__':
    main()