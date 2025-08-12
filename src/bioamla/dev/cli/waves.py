# bioamla/wav_file_list.py
import click
from bioamla.core.batch import get_wav_file_frame
from novus_pytils.txt.csv import write_csv
import sys

@click.command()
@click.argument('input_directory')
@click.argument('output_file')
def main(input_directory, output_file):
    if len(sys.argv) > 1:
        df = get_wav_file_frame(input_directory)
        write_csv(df, output_file)

if __name__ == '__main__':
    main()