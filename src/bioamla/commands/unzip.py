import click
from bioamla.controllers.files import extract_zip_file
 
@click.command()
@click.argument('file_path')
@click.argument('output_path', required=False, default='.')
def main(file_path : str, output_path : str):
  if output_path == '.':
    import os
    output_path = os.getcwd()
  
  extract_zip_file(file_path, output_path)

if __name__ == '__main__':
  main()
  