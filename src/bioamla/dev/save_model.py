"""
Model Serialization Utility

This module provides functionality for loading and saving pretrained AST (Audio Spectrogram Transformer) models.
It allows conversion of models from standard pretrained format to binary serialized format for efficient storage
and deployment.

The module handles model state dictionary serialization and supports the standard Hugging Face model format
for saving pretrained models with all necessary configuration files.
"""

import sys
from typing import Union
from pathlib import Path

def save_model_bin(model_dir: Union[str, Path], bin_filepath: Union[str, Path]) -> None:
    """
    Load a pretrained AST model and save it in binary format.
    
    This function loads a pretrained Audio Spectrogram Transformer model from the specified
    directory and saves both the model state dictionary as a binary file and the complete
    model configuration using the Hugging Face save_pretrained method.
    
    Args:
        model_dir (Union[str, Path]): Path to the directory containing the pretrained model.
                                     This should contain model configuration files and weights.
        bin_filepath (Union[str, Path]): Path where the binary model state dictionary will be saved.
                                        Should have .bin or .pt extension.
    
    Raises:
        ImportError: If required dependencies (torch, transformers) are not available.
        FileNotFoundError: If the model directory does not exist or is invalid.
        RuntimeError: If model loading or saving fails.
    
    Example:
        >>> save_model_bin('./pretrained_ast_model', './model_weights.bin')
        
    Note:
        This function creates two outputs:
        1. A binary file at bin_filepath containing the model state dictionary
        2. Updates the model_dir with saved pretrained model files
    """
    from bioamla.ast import load_pretrained_ast_model
    import torch
    
    # Load the pretrained model from the specified directory
    model = load_pretrained_ast_model(model_dir)
    
    # Save the model state dictionary as a binary file
    torch.save(model.state_dict(), bin_filepath)
    
    # Save the complete pretrained model with configuration
    model.save_pretrained(model_dir)

def main() -> None:
    """
    Command-line interface for model saving functionality.
    
    Expects two command-line arguments:
    1. model_path: Path to the pretrained model directory
    2. bin_filepath: Path for the output binary file
    
    Example usage:
        python save_model.py ./pretrained_model ./model_weights.bin
    """
    if len(sys.argv) != 3:
        print("Usage: python save_model.py <model_path> <bin_filepath>")
        print("  model_path: Path to the pretrained model directory")
        print("  bin_filepath: Path for the output binary file")
        sys.exit(1)
    
    model_path = sys.argv[1]
    bin_filepath = sys.argv[2]
    
    try:
        save_model_bin(model_path, bin_filepath)
        print(f"Successfully saved model binary to: {bin_filepath}")
        print(f"Model configuration saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()