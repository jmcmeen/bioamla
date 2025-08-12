import click
from bioamla.controllers.diagnostics import get_device_info

@click.command()
def main():
    device_info = get_device_info()
    
    click.echo("Devices:")
    click.echo(f'CUDA available: {device_info["cuda_available"]}')
    click.echo(f'Current device: {device_info["current_device"]}')
    click.echo(f'Device count: {device_info["device_count"]}')

    for device in device_info['devices']:
        click.echo(f'  - Index: {device["index"]}, Name: {device["name"]}')

if __name__ == '__main__':
    main()