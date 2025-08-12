from pydub import AudioSegment
import os
from novus_pytils.files import get_file_name, get_file_directory, directory_exists, create_directory

def convert_wav(file_path :str, format : str, extension : str, output_dir : str = None, new_file_name : str = None):
    if output_dir is None:
        output_dir = get_file_directory(file_path)

    if new_file_name is None:
        new_file_name = get_file_name(file_path.replace(".wav", extension))
    else:
        new_file_name = new_file_name + extension

    if not directory_exists(output_dir):
        create_directory(output_dir)

    AudioSegment.from_wav(file_path).export(os.path.join(output_dir, new_file_name), format=format)
