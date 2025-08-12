"""
Device Information Command
=========================

Command-line tool for displaying system device information, particularly
GPU and CUDA availability. This utility helps verify hardware resources
available for machine learning and audio processing tasks.

Usage:
    devices

Examples:
    devices                 # Display all available compute devices

Output includes:
    - CUDA availability status
    - Current active device
    - Total device count
    - Detailed information for each device (index and name)
"""

import click
from bioamla.controllers.diagnostics import get_device_info

@click.command()
def main():
    """
    Display comprehensive device information including CUDA and GPU details.
    
    Retrieves and displays information about available compute devices,
    focusing on CUDA-capable GPUs that can be used for machine learning
    inference and training tasks.
    """
    device_info = get_device_info()
    
    click.echo("Devices:")
    click.echo(f'CUDA available: {device_info["cuda_available"]}')
    click.echo(f'Current device: {device_info["current_device"]}')
    click.echo(f'Device count: {device_info["device_count"]}')

    for device in device_info['devices']:
        click.echo(f'  - Index: {device["index"]}, Name: {device["name"]}')

if __name__ == '__main__':
    main()