from novus_pytils.files import download_file

def download_file_from_url(url: str, output_path: str) -> None:
    """
    Downloads a file from the specified URL to the given output path.
    
    :param url: The URL of the file to download.
    :param output_path: The local path where the downloaded file will be saved.
    """
    download_file(url, output_path)


