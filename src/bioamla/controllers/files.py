from novus_pytils.files import download_file, extract_zip_file

def download_file_from_url(url: str, output_path: str) -> None:
    """
    Downloads a file from the specified URL to the given output path.
    
    :param url: The URL of the file to download.
    :param output_path: The local path where the downloaded file will be saved.
    """
    download_file(url, output_path)

def extract_zip_file_to_directory(zip_file_path: str, output_dir: str) -> None:
    """
    Extracts a ZIP file to the specified output directory.
    
    :param zip_file_path: The path to the ZIP file to extract.
    :param output_dir: The directory where the contents of the ZIP file will be extracted.
    """
    extract_zip_file(zip_file_path, output_dir)
