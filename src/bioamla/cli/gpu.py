import click
import torch

@click.command()
def main():
    cuda_available = torch.cuda.is_available()
    current_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)

    click.echo(f'cuda available: {cuda_available}')
    click.echo(f'current device: {current_device}')
    click.echo(f'device count: {device_count}')
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        click.echo(f'device name: {device_name}')

if __name__ == '__main__':
    main()