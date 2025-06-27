# bioamla/batch.py
import pandas as pd
from novus_pytils.wave import get_wav_files_metadatam, get_wav_files

def get_wav_file_frame(dir: str) -> pd.DataFrame:

    wav_files_metadata = get_wav_files_metadata(get_wav_files(dir))
    df = pd.DataFrame(wav_files_metadata, columns=['filepath', 'file_size', 'num_channels', 'sample_width', 'frame_rate', 'num_frames', 'duration'])
    return df

