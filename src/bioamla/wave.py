import wave
from novus_pytils.files import get_files_by_extension

def get_wav_metadata(wav_filepath : str) -> dict:
    with wave.open(wav_filepath, 'rb') as wav_file:
        return {
            "filepath": wav_filepath,
            "file_size": wav_file.getnframes() * wav_file.getnchannels() * wav_file.getsampwidth(),
            "num_channels": wav_file.getnchannels(),
            "sample_width": wav_file.getsampwidth(),
            "frame_rate": wav_file.getframerate(),
            "num_frames": wav_file.getnframes(),
            "duration": wav_file.getnframes() / wav_file.getframerate()
        }
    
def get_wav_files_metadata(wav_files : list) -> list:
    return [get_wav_metadata(wav_file) for wav_file in wav_files]

def get_wav_files(directory : str) -> list:
    return get_files_by_extension(directory, ['.wav'])
