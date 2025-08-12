"""
File Download Command
====================

Command-line tool for downloading files from URLs to a specified directory.
This utility provides a simple interface for downloading files, with automatic
filename extraction from the URL and support for custom output directories.

Usage:
    download URL [OUTPUT_DIR]

Examples:
    download https://example.com/file.zip            # Download to current directory
    download https://example.com/data.tar.gz ./data  # Download to ./data directory
    download https://site.com/model.bin /models      # Download to /models directory

Features:
    - Automatic filename detection from URL
    - Custom output directory support
    - Progress indication during download
    - Error handling for network issues
"""

import click 

@click.command()
@click.argument('url', required=True)
@click.argument('output_dir', required=False, default='.')
def main(url: str, output_dir: str):
    """
    Download a file from the specified URL to the target directory.
    
    Downloads a file from the given URL and saves it to the specified output
    directory. If no output directory is provided, downloads to the current
    working directory.
    
    Args:
        url (str): The URL of the file to download
        output_dir (str): Directory where the file should be saved.
                         Defaults to current directory if not specified.
    """
    from bioamla.controllers.files import download_file_from_url
    import os
    
    if output_dir == '.':
        output_dir = os.getcwd()
        
    download_file_from_url(url, output_dir)

if __name__ == '__main__':
  main()
  