"""
Global Configurations
=====================

This module loads and stores configuration values from YAML files in the config directory.
Configuration dictionaries are available for use throughout the bioamla package.
"""

import os
from novus_pytils.text.yaml import load_yaml


def _load_config_file(filename: str) -> dict:
    """Load a YAML configuration file from the config directory."""
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    config_path = os.path.join(config_dir, filename)
    return load_yaml(config_path)


# Load AST batch inference configuration
AST_BATCH_INFERENCE_CONFIG = _load_config_file('ast_batch_inference_config.yml')

# Load AST finetune configuration  
AST_FINETUNE_CONFIG = _load_config_file('ast_finetune_config.yml')