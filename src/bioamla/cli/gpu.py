import click
import torch

@click.command()
def main():
    cuda_available = torch.cuda.is_available()
    current_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    click.echo(f"GPU system info:")

    click.echo(f'CUDA available: {cuda_available}')
    click.echo(f'Current device: {current_device}')
    click.echo(f'Device count: {device_count}')

    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        click.echo(f'Device {i}: {device_name}')

if __name__ == '__main__':
    main()