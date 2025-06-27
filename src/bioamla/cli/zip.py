import click
from bioamla.files import download_file, extract_zip_file
from novus_pytils.files import delete_file
 
# test_data_url = "https://www.bioamla.org/datasets/scp_small.zip"

@click.command()
@click.argument('zip_file_url')
@click.argument('output_dir')
def main(zip_file_url, output_dir):
  temp_file = "temp.zip"
  download_file(zip_file_url, temp_file)
  extract_zip_file(temp_file, output_dir)
  delete_file(temp_file)

if __name__ == '__main__':
  main()
  