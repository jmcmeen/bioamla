import click 

@click.command()
@click.argument('url', required=True)
@click.argument('output_dir', required=False, default='.')
def main(url : str, output_dir : str):
    from bioamla.controllers.files import download_file_from_url
    import os
    
    if output_dir == '.':
        output_dir = os.getcwd()
        
    download_file_from_url(url, output_dir)

if __name__ == '__main__':
  main()
  